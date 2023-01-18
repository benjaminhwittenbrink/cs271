import cv2 
from PIL import Image

import torch 

import os 
os.chdir("drive/MyDrive/bmi/")

from transformers import ViTForImageClassification, ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

model = ViTForImageClassification.from_pretrained("LOAD MODEL PATH")
model.eval()

def visualize_attention(model, feature_extractor, img_loc):
  img_np = np.load(img_loc)
  img = Image.fromarray(img_np)
  ft_ex_img = feature_extractor(img_np)['pixel_values'][0]
  output = model(torch.unsqueeze(torch.Tensor(ft_ex_img), 0), output_attentions = True)

  att_mat = torch.stack(output['attentions']).squeeze(1)
  # Average the attention weights across all heads.
  att_mat = torch.mean(att_mat, dim=1)  

  # To account for residual connections, we add an identity matrix to the
  # attention matrix and re-normalize the weights.
  residual_att = torch.eye(att_mat.size(1))
  aug_att_mat = att_mat + residual_att
  aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

  # Recursively multiply the weight matrices
  joint_attentions = torch.zeros(aug_att_mat.size())
  joint_attentions[0] = aug_att_mat[0]

  for n in range(1, aug_att_mat.size(0)):
      joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

  # Attention from the output token to the input space.
  v = joint_attentions[-1]
  grid_size = int(np.sqrt(aug_att_mat.size(-1)))
  mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()


  mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
  result = (mask * img).astype("uint8")

  fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))

  ax1.set_title('Original')
  ax2.set_title('Attention Map')
  ax3.set_title('Attention Mask')
  _ = ax1.imshow(img)
  _ = ax2.imshow(result)
  _ = ax3.imshow(np.squeeze(mask), cmap = 'gray')
  plt.show()


files = os.listdir("west_africa")


visualize_attention(model, feature_extractor, "west_africa/" + files[i])



