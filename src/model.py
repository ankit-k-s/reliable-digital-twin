import torch
import torch.nn as nn

class WADILSTMAutoencoder(nn.Module):
    def __init__(self, num_sensors=127, hidden_dim=64, latent_dim=32):
        super(WADILSTMAutoencoder, self).__init__()
        
        self.num_sensors = num_sensors
        
        # ---------------------------
        # ENCODER
        # ---------------------------
        self.encoder_l1 = nn.LSTM(
            input_size=num_sensors, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        self.encoder_l2 = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=latent_dim, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(p=0.2)
        
        # ---------------------------
        # DECODER
        # ---------------------------
        self.decoder_l1 = nn.LSTM(
            input_size=latent_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        self.decoder_l2 = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=num_sensors, 
            batch_first=True
        )

    def forward(self, x):
        enc, _ = self.encoder_l1(x)
        enc = self.dropout(enc)
        enc, _ = self.encoder_l2(enc)
        
        dec, _ = self.decoder_l1(enc)
        dec = self.dropout(dec)
        reconstruction, _ = self.decoder_l2(dec)
        
        return reconstruction

def compute_residuals(model, x):
    """
    Helper function for Layer 3 (Deviation Detection) and Layer 4 (Reliability).
    Calculates the absolute error between the real sensor data and the twin's prediction.
    """
    model.eval()
    with torch.no_grad():
        reconstruction = model(x)
        # Per-timestep, per-sensor absolute error
        residuals = torch.abs(x - reconstruction)
        
        # Collapse sensors -> single scalar per timestep 
        # This gives us the overall system deviation at time 't'
        r_t = residuals.mean(dim=2)  # shape: [batch, 60]
        
    return r_t

if __name__ == "__main__":
    # Test the architecture
    dummy_batch = torch.randn(128, 60, 127) 
    model = WADILSTMAutoencoder(num_sensors=127)
    output = model(dummy_batch)
    
    print(f"Input shape:  {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")