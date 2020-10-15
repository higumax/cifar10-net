import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3), # 32x32x16                 
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3), # 16x16x32                
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # 16x16x64               
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),                                                      
        )


        self.decoder = nn.Sequential(                                                     
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3), # 16x16x32                
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3), # 16x16x32                                                                                      
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3), # 32x32x3                                                                                       
            nn.BatchNorm2d(num_features=3),
            nn.Tanh(),
        )


    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return latent, output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), ## 16x16x64               
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), ## 14x14x64               
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5), ## 10x10x64               
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1), ## 10x10x128             
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3), ## 8x8x128              
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5), ## 4x4x128              
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), ## 2x2x256              
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * 256, 10),
        )


        self.autoencoder = AutoEncoder()


    def forward(self, x):
        latent, autoencoder_output = self.autoencoder(x)
        tmp = self.cnn(latent)
        tmp = tmp.view(tmp.size(0), -1)  # flatten
        output = self.fc(tmp)
        return output, autoencoder_output

if __name__ == "__main__":
    model = Classifier()
    print(model)