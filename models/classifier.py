
import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        #super(Classifier, self).__init__()
        self.num_classes = num_classes
        # Layer Definitions
        #self.fc1 = torch.nn.Linear(256*8*8, self.num_classes)
        self.fc1 = torch.nn.Linear(256*8*8, 1024)
        self.fc2 = torch.nn.Linear(1024,512)
        self.fc3 = torch.nn.Linear(512,512)
        self.fc4 = torch.nn.Linear(512, self.num_classes)

    # image_features is of dimensions [batch_size, n_views, 256, 8, 8]
    # the output is the class scores for each view; in total, batch_size * n_views many scores
    def forward(self, image_features):
        # contigious forces a copy from the permute (which would ordinarily just change metainfo)
        #image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()

        num_batches = image_features.shape[0]
        n_views = image_features.shape[1]

        # (batch_size * n_views, 256,8,8)
        raw_class_scores = image_features.reshape(-1,256,8,8)

        # (batch_size * n_views, 256*8*8)
        raw_class_scores = raw_class_scores.view(-1, 256*8*8)
        raw_class_scores = F.relu(self.fc1(raw_class_scores))
        raw_class_scores = F.relu(self.fc2(raw_class_scores))
        raw_class_scores = F.relu(self.fc3(raw_class_scores))
        raw_class_scores = F.relu(self.fc4(raw_class_scores))

        # (batch_size, n_views, self.num_classes)
        raw_class_scores = raw_class_scores.reshape(num_batches, n_views, self.num_classes)

        return raw_class_scores