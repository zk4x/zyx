
## User

(softmax, but in a dumb way :D)

```rust
let x = dev.randn(1024);
let y = x.exp()/x.exp().sum(..);
```


## AST

1. Create standard ASTs
2. add loops and binds to views
3. merge them all together into one big AST


0 loop global 1, bind 0
1 define accumulator
2 loop register 1024, bind 1
3 move global register id 0 view contiguous
4 exp 1
5 add 2, 4
6 end loop
7 move register global id 5 view contiguous
8 end loop

0 loop global 1024, bind 0
1 move global register id 0 view contiguous
2 move global register id 1 view expanded
3 div 1, 2
4 move register global id 3 view contiguous
5 end loop



## IR



## OpenCL


```c
unsigned int rid0 = get_group_size(0); /* 0..1024 */
if (rid0 < 1) {
    float rmem0 = 0.0f;
    for (unsigned int rid1 = 0; rid1 < 1024; rid1++) {
        float rmem1 = gmem1[rid0*1024+rid1];
        float rmem2 = exp(rmem1);
        rmem0 = rmem0 + rmem2;
    }
    gmem0[rid0] = rmem0;
}
barrier(GLOBAL_MEM_FENCE);
float rmem0 = gmem1[rid0];
float rmem1 = exp(rmem0);
rmem0 = gmem0[0]:
float rmem2 = rmem1 / rmem0;
gmem2[rid0] = rmem2;
```

