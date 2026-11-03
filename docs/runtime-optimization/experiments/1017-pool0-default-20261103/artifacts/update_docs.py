from pathlib import Path
import json
R = Path('/home/sergei/Github/llama.cpp')
N = Path('/home/sergei/_my_old/_my_deepseek41')
s = json.loads(Path('summary.json').read_text())
assert s['complete']
link = 'experiments/1017-pool0-default-20261103/REPORT.md'
note = '''

## Текущие candidate defaults: native gate закрыт

[Отчёт](LINK): runtime `my_maya 0e6f3cec3` и defaults не менялись. Pooling не задан (OFF); в default arm BLOCK_LOCAL также не задан (ON). На двух 98K text/code корпусах 24 группы logits/taps/state/rollback совпали побитово, delta NLL=0; четыре seed failures/retry прошли. Native OFF/default/OFF: 27 output arrays и draft counters совпали. PP98K 1107.75 -> 1138.16 t/s (+2.75%); TG 46.21 -> 45.82 (-0.84%, по одному измерению на arm, ускорение TG не установлено).

Actual ctx=163840 / prefix=159744 / repeat 8K прошёл: минимум 679.25 MiB свободной VRAM, все 15360 экспертных bundles на GPU, placement совпал, exit 0. Context=176128 не допущен расчётом при pooling OFF; прежний actual 176128 с pooling=1 остаётся отдельной серией. Это fixed Q3/MXFP8 Engram/Q8 KV/BF16 DSpark K3/P1 конфигурация, не универсальный предел контекста.

Ошибка post-test сборки сводки (`queue.py` перекрыл stdlib) вызвала повтор metadata. Исходные precision/native/payload manifests сохранены побайтно; память и routes metadata сверены с исходными admission inputs. Incomplete manifest, старые hashes и recovery diagnostics сохранены в отчёте. Runner переименован и защищён от выполнения при импорте.

Далее: matched Maya comparison, свежий residual profile, затем Q2/Q4 MMQ. Media, parallel>1 и auto-fit 262K остаются отдельными проверками. Нового commit/deploy нет.
'''
for name,prefix in [('docs/runtime-optimization/README.md',''),('docs/runtime-optimization/CATALOG.md',''),('docs/runtime-optimization/models/deepseek41-1017.md','../'),('readme_dsv41.md','docs/runtime-optimization/')]:
 p=R/name;t=p.read_text()
 if name.endswith('/README.md'):
  t=t.replace('runtime `b3377774b` плюс проверенный block-local delta с defaultON (pooling остаётся OFF)', 'runtime `0e6f3cec3` (block-local default ON, pooling остаётся OFF)')
  t=t.replace('pool0native/media/parallel>1/auto-fit262K не закрыты этим набором.', 'pool0 native на момент той серии ещё не был закрыт; новый gate приведён ниже. Media/parallel>1/auto-fit262K остаются отдельно.')
 p.write_text(t+note.replace('LINK',prefix+link))
p=R/'docs/runtime-optimization/LESSONS.md'
p.write_text(p.read_text()+'''

## Проверять defaults нужно и в самом стенде

[Pool0 native gate](experiments/1017-pool0-default-20261103/REPORT.md): убрать переменную из launcher недостаточно, если helper задаёт её внутри corpus loop. Проверять child environment, source и actual graph owners. Значение, фиксируемое конструктором, сравнивать в свежих процессах. Успех memory gate при pooling=1 не переносится на pooling OFF: меняется per-device reserve; допуск считать по каждой GPU и подтверждать actual запуском.

Не давать исполняемым Python harness имена stdlib modules (`queue.py`). Защищать запуск `if __name__ == '__main__'`. Повтор metadata при сборке итогов может перезаписать manifests: сохранить диагностику и исходные hashes, проверить независимые admission inputs, отдельно назвать невосстановимые bytes. Не исправлять raw incomplete manifest на complete. Near-prefix другого размера без token oracle является memory/lifecycle gate, не доказательством качества.
''')
for name,rel in [('docs/runtime-optimization/experiments/1017-candidate-block-local-20261103/REPORT.md','../1017-pool0-default-20261103/REPORT.md'),('docs/runtime-optimization/designs/1017-candidate-block-local.md','../experiments/1017-pool0-default-20261103/REPORT.md')]:
 p=R/name;p.write_text(p.read_text()+'\n\n## Позднейшее закрытие pool0 native gap\n\n[Отдельный gate]('+rel+') на `0e6f3cec3` проверил actual defaults: 24 raw groups exact, delta NLL=0, четыре seed failures/retry, 27 native requests и actual ctx=163840 с минимумом 679.25 MiB. Исходные pool=1 gates, скорость и actual 176128 этого отчёта остаются историческими результатами. Runtime/defaults не менялись.\n')
for name in ['DEEPSEEK_V41_1017_MAYA_CACHE_STAGE5_DESIGN.md','DEEPSEEK_V41_1017_MAYA_EXPERT_CACHE_PLAN.md']:
 p=N/name;p.write_text(p.read_text()+note.replace('LINK',str(R/'docs/runtime-optimization'/link)))
d=N/'validation/moe-cache/1017-pool0-default-20261103';d.mkdir(parents=True,exist_ok=True)
(d/'REPORT.md').write_text('# Текущие candidate defaults\n\n[Полный отчёт]('+str(R/'docs/runtime-optimization'/link)+').\n\nPool OFF / block default ON: точность, rollback, native speed и actual memory проверены. Runtime/defaults не менялись. Границы и восстановление metadata после ошибки стенда описаны в canonical отчёте.\n')
