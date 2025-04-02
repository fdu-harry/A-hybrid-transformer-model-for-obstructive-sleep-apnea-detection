# Import various deep learning models for 1D signal processing
from Hybrid_Net import Transformer

warnings.filterwarnings('ignore',category=UserWarning,module='matplotlib.font_manager')

# Setting global parameters
batch_size = 512  # Number of samples processed in each batch
d_model = 128     # Model dimension 
d_inner = 512     # Inner dimension of feed forward network
num_layers = 3    # Number of transformer layers
num_heads = 4     # Number of attention heads
class_num = 2     # Number of output classes
dropout = 0.0     # Dropout rate
warm_steps = 4000 # Steps for learning rate warmup
num_epochs = 200  # Total training epochs
SIG_LEN = 256    # Input signal length
ecg_lead = 3     # Number of input channels
feature_attn_len = 64  # Feature attention length

# Setting up GPU/CPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initializing and loading models
# All models below expect input shape of (batch_size, 3, 256)
# where:
# - batch_size: number of samples in a batch 
# - 3: number of input channels
# - 256: length of 1D signal sequence

model = Transformer(device=device, d_feature=SIG_LEN, d_model=64, d_inner=d_inner,
            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

# Move model to specified device (GPU/CPU)
model = model.to(device)
