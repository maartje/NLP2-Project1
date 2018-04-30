def has_converged_AER(i, AER, prevAER, LLood, prevLLhood):
    return prevAER - AER < 0

def has_converged_LLhood(i, AER, prevAER, LLood, prevLLhood):
    return abs(LLood  - prevLLhood)/abs(prevLLhood) < 0.01