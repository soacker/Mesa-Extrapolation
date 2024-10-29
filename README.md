

## Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs

Implementation of the proposed Mesa-Extrapolation in [Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs](https://arxiv.org/pdf/2410.15859).

### 1.Abstract
Large language models (LLMs), although having revolutionized many fields, still suffer from the challenging extrapolation problem, where the inference ability of LLMs sharply declines beyond their max training lengths. 
In this work, we conduct a theoretical analysis to better understand why No Position Encoding (NoPE) fails outside its effective range, as well as examining the power of Position Encoding (PE) in this context. Our findings reveal that with meticulous weave position, PE can indeed be extended beyond effective range.
Our theorems establish that LLMs equipped with weave PE can achieve improved extrapolation performance without additional cost.
Furthermore, we introduce a novel weave PE method, Mesa-Extrapolation, which utilizes a chunk-based triangular attention matrix and applies Stair PE to manage the final chunk.
This method not only retains competitive performance but also offers substantial benefits such as significantly reduced memory demand and faster inference speed. Extensive experiments validate the effectiveness of Mesa-Extrapolation, demonstrating its potential as a scalable solution to enhancing LLMs' applicative reach.


### 2.Overall

The schematic diagram of our method is shown below：

<div style="text-align: center;">
  <img src="img.png" alt="Image 1" width="1000"/>
</div>

Our approach achieves minimal memory usage and the fastest inference latency：

<div style="text-align: center;">
  <img src="img_1.png" alt="Image 2" width="800"/>
</div>

It extends the existing Phi-3-instruct model, which supports a sequence length of 128k, to at least 192k：
<div style="text-align: center;">
  <img src="img_2.png" alt="Image 3" width="800"/>
</div>


### 3.Usage
#### Dependencies
Our current implementation is based on transformers==4.31.0. We will continue to update it in the future. 
For attention calculation, we currently support both the flash-attention and torch implementation.

#### Passkey Data Generation
`python datas/make_passkey_data.py`

#### Run
`python experiments/evaluate_passkey_retrieval.py`


### 4.TODOs
[1] Release core code of Mesa-Extrapolation, including llama, Pythia, Baichuan, Phi, etc.

[2] Supports implementation with newer versions of Transformers.

[3] Integrates with open-source inference frameworks such as vLLM.

### 5.Contributing
We welcome contributions from the research community to improve the efficiency of MesaExtrapolation. If you have any idea or would like to join in, please contact us (xin.ma0206@gmail.com).

If you find our method useful, please kindly cite our paper.

```
@misc{xin2024llm,
      title={Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs}, 
      author={Xin Ma and Yang Liu and Jingjing Li and Xiaoxu Ma},
      year={2024},
      eprint={2410.15859},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```




