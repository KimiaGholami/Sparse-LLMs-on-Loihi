# scripts/eval_capo.py
import sys, os, json, math, torch
os.chdir('/mnt/cephfs/share/kimia/flame')
sys.path.insert(0, '.')

import custom_models.gated_deltaproduct
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('fla-hub/transformer-1.3B-100B')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_loss_on_file(model, filepath, max_samples=500):
    model.eval()
    total_loss = 0.0
    count = 0
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()][:max_samples]
    with torch.no_grad():
        for line in lines:
            inputs = tokenizer(line, return_tensors='pt',
                               truncation=True, max_length=512).to(device)
            if inputs['input_ids'].shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')

def compute_bits_per_param(loss, N_people, num_params):
    max_loss = math.log(32000)
    frac_memorized = max(0, 1.0 - loss / max_loss)
    bits = N_people * 47.6 * frac_memorized
    return bits / num_params

DATA = '/mnt/cephfs/share/kimia/benchmark_data/capo'
EXP  = '/mnt/cephfs/share/kimia/flame/exp/capo'

results = {}

for N in [20000, 50000, 100000, 200000, 500000]:
    for model_name in ['gated_deltanet', 'gated_deltaproduct']:
        key = f'{model_name}_N{N}'
        # Use the HF-converted path (produced by convert_dcp_to_hf)
        ckpt_path = f'{EXP}/{model_name}/N{N}'
        test_file = f'{DATA}/N{N}/test.txt'

        print(f'\nLoading {key}...')
        try:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16   # required for GatedDeltaProduct
            ).to(device)
            num_params = sum(p.numel() for p in model.parameters())
            loss = compute_loss_on_file(model, test_file)
            bits = compute_bits_per_param(loss, N, num_params)
            results[key] = {'loss': loss, 'bits_per_param': bits,
                            'num_params': num_params, 'N': N}
            print(f'  loss={loss:.4f}  bits/param={bits:.4f}  params={num_params:,}')
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'  ERROR: {e}')
            results[key] = {'error': str(e)}

# Print comparison table
print('\n' + '='*70)
print(f'{"N":>8}  {"GatedDeltaNet":>15}  {"GatedDeltaProduct":>18}  {"Winner":>12}')
print(f'{"":>8}  {"bits/param":>15}  {"bits/param":>18}')
print('='*70)
for N in [20000, 50000, 100000, 200000, 500000]:
    dn = results.get(f'gated_deltanet_N{N}', {}).get('bits_per_param', float('nan'))
    dp = results.get(f'gated_deltaproduct_N{N}', {}).get('bits_per_param', float('nan'))
    winner = 'tie'
    if dn > dp + 0.001: winner = 'DeltaNet'
    elif dp > dn + 0.001: winner = 'DeltaProduct'
    print(f'{N:>8}  {dn:>15.4f}  {dp:>18.4f}  {winner:>12}')

os.makedirs('results', exist_ok=True)
with open('results/capo_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to results/capo_results.json')