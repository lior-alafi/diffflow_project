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

SaPE²:
q,k
 →
g_x , g_y
 →
p_x , p_y
 →
q·e_x , q·e_y
 →
interpolate using p_x , p_y
 →
SaPE_x ∈ R^W , SaPE_y ∈ R^H
 →
||SaPE_x[i] - SaPE_x[j]||₂
+
||SaPE_y[i] - SaPE_y[j]||₂
 →
attention logits + SaPE bias
 →
softmax