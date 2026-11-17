import sys
sys.path.insert(0, r'C:\Users\Tolulope Olugbemi\.copilot\repos\Barca-FC-MTAH\evaluation')
import possession_metrics as pm

gt_df = pm.read_label_csv(r'C:\Users\Tolulope Olugbemi\.copilot\repos\Barca-FC-MTAH\BAR-ATH_possession_gt_public.csv')
pred_df = pm.read_label_csv(r'C:\Users\Tolulope Olugbemi\.copilot\repos\Barca-FC-MTAH\haeys\out\Vector.csv')

gt = pm.label_codes(gt_df)
pred = pm.label_codes(pred_df)

result = pm.evaluate(gt, pred, bootstrap_n=1000)

print('=== VECTOR (haeys) vs PUBLIC GT ===')
print(f'macro_f1: {result["macro_f1"]:.4f}')
print(f'spell_f1_mean (RANK): {result["spell_f1_mean"]:.4f}')
print(f'spell_f1@0.5: {result["spell_f1_at_headline"]:.4f}')
print(f'alive_dead_f1: {result["alive_dead_f1"]:.4f}')
print(f'accuracy: {result["accuracy"]:.4f}')
print(f'home_share_err: {result["share_err"]:.4f}')
print(f'gt_spells: {result["n_gt_spells"]}, pred_spells: {result["n_pred_spells"]}')
print(f'gt_mean_dur: {result["gt_mean_spell_duration_s"]:.1f}s, pred_mean_dur: {result["mean_spell_duration_s"]:.1f}s')
print(f'transitions@12: P={result["transitions"][12]["precision"]:.3f} R={result["transitions"][12]["recall"]:.3f} F1={result["transitions"][12]["f1"]:.3f}')
print(f'transitions@25: P={result["transitions"][25]["precision"]:.3f} R={result["transitions"][25]["recall"]:.3f} F1={result["transitions"][25]["f1"]:.3f}')
print(f'bootstrap spellF1: [{result["spell_bootstrap_low"]:.3f}, {result["spell_bootstrap_high"]:.3f}]')
print(f'bootstrap macroF1: [{result["bootstrap_low"]:.3f}, {result["bootstrap_high"]:.3f}]')