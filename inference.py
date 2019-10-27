from inference_helpers import *

model = None
model = load_model("model.h5")

#model.summary()

img = get_data("dr.jpg")
#print(img.shape)

class_to_name = ["Normal", "Diabetic Retina"]

print("Predicting...")
pred, pred_arg = inference(img, model)
#print(pred, pred_arg)

print(class_to_name[pred_arg])


