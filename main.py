from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
import dspy
from dspy.evaluate import answer_exact_match
from dspy.utils.dummies import DummyLM
from typing import Dict, Any, List, Optional, Type
from itertools import cycle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="DSPy Proxy Server")

# In-memory storage
signatures: Dict[str, Any] = {} # Stores Signature classes
compiled_modules: Dict[str, Any] = {} # Store optimized modules

class RegisterRequest(BaseModel):
    name: str
    signature: str  # e.g., "question -> answer"
    instructions: Optional[str] = None

class ConfigureRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.0

class PredictRequest(BaseModel):
    signature_name: str
    inputs: Dict[str, Any]
    module_type: str = "Predict"  # Predict, ChainOfThought, ReAct, etc.
    compiled_module_id: Optional[str] = None # If using a previously optimized module

class OptimizeRequest(BaseModel):
    signature_name: str
    train_data: List[Dict[str, Any]] # List of input/output pairs
    metric: str = "exact_match" # exact_match, etc.
    optimizer: str = "BootstrapFewShot"
    max_bootstraps: int = 4

@app.post("/configure")
def configure_lm(req: ConfigureRequest):
    if req.provider == "dummy":
        # Simple dummy that returns a fixed response or rotates through a list
        # ChainOfThought expects 'reasoning' field usually
        # Multiply list to simulate infinite responses for testing
        lm = DummyLM([{"answer": "42", "reasoning": "because"}] * 1000)
        dspy.settings.configure(lm=lm)
        return {"status": "configured", "model": "dummy"}

    # Determine API Key
    api_key = req.api_key
    if not api_key:
        # Try to find it in env vars based on provider
        env_var_name = f"{req.provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)
        
        # Fallback for openai if provider is just 'openai'
        if not api_key and req.provider == "openai":
             api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key required for provider '{req.provider}'. Please provide it in the request or set {req.provider.upper()}_API_KEY environment variable.")

    # Construct Model String for dspy.LM (litellm)
    # dspy.LM expects "provider/model" usually, or just "model" if it's openai
    model_name = req.model
    if "/" not in model_name:
        if req.provider != "openai":
             model_name = f"{req.provider}/{req.model}"
        else:
             # For OpenAI, dspy/litellm often handles just "gpt-4" etc, but "openai/gpt-4" is safer
             model_name = f"openai/{req.model}"

    try:
        lm = dspy.LM(model=model_name, api_key=api_key, max_tokens=req.max_tokens, temperature=req.temperature)
        dspy.settings.configure(lm=lm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure LM: {str(e)}")
    
    return {"status": "configured", "model": model_name, "provider": req.provider}

@app.post("/register")
def register_signature(req: RegisterRequest):
    try:
        # Create a dynamic signature class
        sig = dspy.make_signature(req.signature, instructions=req.instructions or "")
        signatures[req.name] = sig
        return {
            "status": "registered", 
            "name": req.name, 
            "fields": list(sig.fields.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest):
    if req.signature_name not in signatures:
        raise HTTPException(status_code=404, detail=f"Signature '{req.signature_name}' not found")
    
    sig = signatures[req.signature_name]
    
    try:
        # If a compiled module is requested
        if req.compiled_module_id:
            if req.compiled_module_id not in compiled_modules:
                raise HTTPException(status_code=404, detail="Compiled module not found")
            module = compiled_modules[req.compiled_module_id]
        else:
            # Instantiate fresh module
            if req.module_type == "Predict":
                module = dspy.Predict(sig)
            elif req.module_type == "ChainOfThought":
                module = dspy.ChainOfThought(sig)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown module type: {req.module_type}")
        
        # Execute
        result = module(**req.inputs)
        
        # Convert Prediction to dict
        output = {}
        for key in sig.fields:
            if hasattr(result, key):
                output[key] = getattr(result, key)
        
        # Also capture rationale if CoT
        if hasattr(result, "rationale"):
            output["rationale"] = result.rationale

        return output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    if req.signature_name not in signatures:
        raise HTTPException(status_code=404, detail="Signature not found")
    
    sig = signatures[req.signature_name]
    
    # Prepare Data
    # Identify input keys from signature
    input_keys = [k for k, v in sig.fields.items() if v.json_schema_extra['__dspy_field_type'] == 'input']
    
    trainset = [dspy.Example(**x).with_inputs(*input_keys) for x in req.train_data]
    
    # Define Metric
    if req.metric == "exact_match":
        def metric_fn(example, pred, trace=None):
            return answer_exact_match(example, pred)
    else:
        # Default to simple equality of the last output field
        output_keys = [k for k, v in sig.fields.items() if v.json_schema_extra['__dspy_field_type'] == 'output']
        output_field = output_keys[-1]
        def metric_fn(example, pred, trace=None):
            return getattr(example, output_field) == getattr(pred, output_field)

    try:
        # Optimizer
        if req.optimizer == "BootstrapFewShot":
            optimizer = dspy.BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=req.max_bootstraps)
        else:
            raise HTTPException(status_code=400, detail="Only BootstrapFewShot supported for now")
        
        # Compile
        student = dspy.ChainOfThought(sig)
        
        compiled_program = optimizer.compile(student, trainset=trainset)
        
        # Store compiled program
        module_id = f"{req.signature_name}_opt_{len(compiled_modules)}"
        compiled_modules[module_id] = compiled_program
        
        return {"status": "optimized", "module_id": module_id}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
