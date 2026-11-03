from pathlib import Path
import shutil
repo = Path('/home/sergei/Github/llama.cpp')
name = '1017-candidate-block-local.md'
shutil.copyfile(Path('/tmp/dsv41-candidate-block-local-20261103') / name, repo / 'docs/runtime-optimization/designs' / name)
p = repo / 'docs/runtime-optimization/README.md'
s = p.read_text()
s = s.replace('runtime `c3c91dda8` плюс проверенный candidate-pool opt-in delta (пока без нового коммита)', 'runtime `b3377774b` (candidate-pool opt-in/default OFF)')
s += '''\n\n## Следующий этап: block keep transport\n\nПредыдущий эксперимент закоммичен в `b3377774b`, defaults прежние. Подготовлен [дизайн локального разворачивания candidate mask](designs/1017-candidate-block-local.md): передавать существующий F32 block keep, повторять прежний REPEAT/reshape/trim на GPU-потребителях, сохранять scheduler events и copy slots. Это предложенный следующий этап, не реализованный режим и не новый speed claim. До реализации требуется review изменения graph representation и ownership; план проверок включает exact logits/state/rollback, copy proof, память >=512MiB на каждой GPU и clean PP/TG A/B.\n'''
p.write_text(s)
p = repo / 'docs/runtime-optimization/CATALOG.md'
s = p.read_text().replace('`c3c91dda8` + candidate-pool opt-in delta (uncommitted)', '`b3377774b` (candidate-pool opt-in/default OFF)')
s += '''\n\n### Следующий proposed этап после O11\n\n[Block keep transport/local expansion](designs/1017-candidate-block-local.md): переносить F32 0/-inf по блокам и разворачивать маску на каждом actual consumer device. Design review pending; runtime не изменён, нового флага в сборке ещё нет. Сохранить current defaults, graph reuse key, protected copy slots, fallback и padding strides; не заменять candidate blocks финальным source TopK.\n'''
p.write_text(s)
p = Path('/home/sergei/_my_old/_my_deepseek41/DEEPSEEK_V41_1017_MAYA_CACHE_STAGE5_DESIGN.md')
s = p.read_text()
s += '''\n\n### 2026-11-03: следующий этап - block keep transport/local expansion\n\n- [x] Предыдущий producer-local candidate pool закоммичен: runtime `b3377774b`, notebook `70ab4f2`. Его default остаётся OFF; HC-sharing и другие defaults прежние.\n- [x] Подготовлен отдельный дизайн: переносить существующий F32 `candidate_keep` (0/-inf на блок), разворачивать прежними REPEAT/reshape/trim на actual GPU-потребителях. Учитываются graph reuse, allocator lifetime, n_copies/cur_copy, события, padded strides и fallback; новые kernels/streams/scheduler alias не входят в этап.\n- [ ] Review дизайна перед изменением graph representation/ownership. Новый предполагаемый `LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1` пока не реализован и не доступен в runtime.\n- [ ] После review: bounded opt-in prototype/default OFF, operator/capture/slot rotation, negative guards, exact target/draft/logits/state/rollback/NLL, metadata и actual memory с floor512MiB, pointer copy proof, clean OFF/ON/OFF PP512/8K/98K и TG256. При нехватке VRAM уменьшить контекст.\n- [ ] Дальше отдельно matched Maya и Q2/Q4; загрузку CPU Maya не считать доказательством CPU expert arithmetic. В fast73.8t/s логе CPU lane0.0%, cache hit100%; профиль CPU time/token остаётся отдельной возможной диагностикой. Новых GPU-тестов этого этапа пока нет.\n\n[Дизайн в my_maya](/home/sergei/Github/llama.cpp/docs/runtime-optimization/designs/1017-candidate-block-local.md).\n'''
p.write_text(s)
