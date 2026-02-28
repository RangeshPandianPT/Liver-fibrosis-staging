import timm
import sys
import os

print(f"Python: {sys.version}")
print(f"Timm: {timm.__version__}")
print("Attempting to create model 'deit_small_patch16_224' with pretrained=True...")

try:
    # Set timeout for download? Timm uses torch.hub which uses urllib/requests
    # We can't easily set timeout but we can see if it throws
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    print("✅ Model created successfully!")
except Exception as e:
    print(f"❌ Error creating model: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
