import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Example usage
input_size = 256  # Size of the character vocabulary
hidden_size = 256
output_size = input_size
model = CharRNN(input_size, hidden_size, output_size)

# Training and evaluation would involve:
# - Preparing the training data (text corpus)
# - Creating a vocabulary and encoding characters
# - Defining loss function, optimizer, and training loop
# - Generating text using the trained model

