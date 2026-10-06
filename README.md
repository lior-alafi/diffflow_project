SaPE2
https://arxiv.org/html/2505.09466v1
RoPE
https://arxiv.org/pdf/2104.09864
CoPE
https://arxiv.org/pdf/2405.18719


APE:
x → x + PE

RoPE:
q,k → rotate(q,k)

CoPE:
q,k
 → 
attention logits
 → 
gates
 → 
contextual positions
 → 
positional logits
 → 
attention logits + positional logits
 → 
softmax