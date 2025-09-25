 # Overview 

 - Define keys
```
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # if using Claude
```
- Install
```
pip install -r human-eval/requirements.txt 
python -m simple-evals.simple_evals --list-models
```
- Quick test:
`python -m simple-evals.simple_evals --model=gpt-4.1 --eval=mmlu --examples=10`