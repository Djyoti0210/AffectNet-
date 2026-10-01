import torchvision.transforms as T
import inspect

ra = T.RandAugment()

try:
    if hasattr(ra, '_augmentation_space'):
        op_meta = ra._augmentation_space(10, (224, 224))
        op_names = list(op_meta.keys())
    else:
        op_names = "Could not find augmentation space directly."
except Exception as e:
    op_names = str(e)

print(f"Available ops ({len(op_names) if isinstance(op_names, list) else 0}):")
print(op_names)

print("\n--- RandAugment.forward source ---")
try:
    source_lines = inspect.getsource(ra.forward)
    print(source_lines)
except Exception as e:
    print(e)
