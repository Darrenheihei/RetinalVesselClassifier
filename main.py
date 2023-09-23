from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def predict_model(model, x_val):
    val_preds = model.predict(x_val, batch_size=16)
    return val_preds


def contrast_stretch(x):
  I_max = np.max(x, axis=(1, 2))[:, :, None, None] # shape: (# of sample, 1) -> (# of sample, 1, 1, 1)
  I_min = np.min(x, axis=(1, 2))[:, :, None, None] # shape: (# of sample, 1) -> (# of sample, 1, 1, 1)
  x_enhanced = ((x - I_min)/(I_max - I_min) * 255).astype(int)
  return x_enhanced


def rescale_01(x):
  x_01 = x/255
  return x_01.astype(float)


def threshold(val_preds, thresh_value):
    val_preds_thresh = val_preds >= thresh_value
    return val_preds_thresh.astype(int)


def visualize_side_by_side(img1, img1caption,
                            img2, img2caption, 
                            img2range = (0,255), img1range = (0,255)):
  
    fig, axs = plt.subplots(1,2, figsize=(8,5))
    axs[0].imshow(img1,cmap='gray',vmax=img1range[1],vmin=img1range[0])                       
    axs[1].imshow(img2,cmap='gray',vmax=img2range[1],vmin=img2range[0])                            
    axs[0].set_title(img1caption)     
    axs[1].set_title(img2caption)
    plt.show()


def main():
  retinal_vessel_data = np.load('retinal_vessel_dataset.npz')
  x_val_raw = retinal_vessel_data["x_val"][...,np.newaxis]
  y_val = retinal_vessel_data["y_val"][...,np.newaxis].astype(int)

  x_val_enhanced = contrast_stretch(x_val_raw)
  x_val = rescale_01(x_val_enhanced)

  model = load_model('trainedModel.h5')

  val_preds = predict_model(model, x_val)
  val_preds_thresh = threshold(val_preds, 0.5)  
  
  # allow user to view different results
  print("----------------------------\nClassification has finished.")
  while True:
    samples = input("""
If you wish to view the result of a particular/multiple images, you may input their indexes (from 0 to 172), with each index separated by a comma.
e.g. 0, 10, 22, 131
If you wish to leave, please enter 'q'.
Your choice(s): """)
    if samples == 'q':
      exit()

    try: # in case of receiving wrong input
      samples = map(int, samples.split(', '))
      for i in samples:
        if 0 <= i <= 172:
          visualize_side_by_side(x_val[i,...], 'Input Image', val_preds_thresh[i,...], 'Output Image', (0,1),(0,1))
    except:
      print("Sorry, there is an input error. Please try it again.")
    

      


if __name__ == "__main__":
  main()