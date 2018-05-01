def select_model_AER(AERs):
    return _find(
        lambda i: has_converged_AER(AERs[i-1], AERs[i]), range(1, len(AERs))
    )

def select_model_LLhood(LLhoods):
    return _find(
        lambda i: has_converged_LLhood(LLhoods[i-1], LLhoods[i]), range(1, len(LLhoods))
    )
    
def _find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
        if f(item): 
            return item

def has_converged_AER(prevAER, AER):
    return prevAER - AER < 0

def has_converged_LLhood(prevLLhood, LLood):
    return abs(LLood  - prevLLhood)/abs(prevLLhood) < 0.001
