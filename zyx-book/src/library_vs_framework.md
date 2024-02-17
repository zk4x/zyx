# Library vs. Framework

Zyx aspires to be just a library, not a framework. The difference is that libraries plug into your workflow, while frameworks impose a certain workflow on you.

Some ML libraries force you to do the training loop in their way, sometimes forcing you to statically define the whole graph beforehand and then just call .train() or .fit() to run the whole training loop. This method discourages debugging and makes trial and error development difficult. Because of this, dynamic PyTorch is the most used ML library these days.

Zyx aspires to be even more dynamic than PyTorch, because it does not require you to specify which tensors require gradient beforehand. Instead, you specify which gradients you want to calculate when you call backward function.

