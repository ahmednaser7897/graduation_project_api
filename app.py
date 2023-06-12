from flask import Flask, request, render_template, jsonify
import tensorflow.keras as keras
from test import *


keras.utils.get_custom_objects()['weighted_binary_crossentropy'] = weighted_binary_crossentropy

def predict_old_model(mainImage):
    print("predict_old_model")
    # Generate predictions on test data
    predictions = old_model.predict(mainImage)
    print("3 : change_threshold")
    predictions = change_threshold(predictions)
    print("4 : predictions.shape")
    print(predictions.shape)
    image = get_img_from_array(predictions[0, :, :, 0])
    print("5 : image type")
    print(image)
    # image.save("geeks.png")
    image = im_2_b64(image)
    print("6 : im_2_b64 ")
    print(image)
    return image

def predict_new_model(mainImage):
    print("predict_new_model")
    # Generate predictions on test data
    mainImage = np.transpose(mainImage, (0, 3, 1, 2))
    mainImage = torch.from_numpy(mainImage).float()
    predictions = new_model.predict(mainImage)
    print("3 : change_threshold")
    predictions = change_threshold(predictions)
    print("4 : predictions.shape")
    predictions = np.transpose(predictions, (0, 2, 3, 1))
    print(predictions.shape)
    image = get_img_from_array(predictions[0, :, :, 0])
    print("5 : image type")
    print(image)
    # image.save("geeks.png")
    image = im_2_b64(image)
    print("6 : im_2_b64 ")
    print(image)
    return image


new_model = load_new_model()
#old_model = load_old_model()

app = Flask(__name__)


@app.route('/predictApi', methods=["POST"])
def api2():
    # Get the image from post request
    try:
        print("0 : request.files")
        for x in request.files:
            print(x)
        if 'image1' not in request.files:
            return jsonify({'Error': "Please try again.  Image 1 doesn't exist"})
        if 'image2' not in request.files:
            return jsonify({'Error': "Please try again.  Image 2 doesn't exist"})
        image1 = request.files.get('image1')
        image2 = request.files.get('image2')
        frame1 = get_frame_img(image1)
        frame2 = get_frame_img(image2)
        mainImage = map_filename_to_image_and_mask(frame1, frame2)
        #old_model_predictions = predict_old_model(mainImage)
        new_model_predictions = predict_new_model(mainImage)
        return jsonify({'new_model_predictions': new_model_predictions.decode("utf-8")})
        #return jsonify({'new_model_predictions': new_model_predictions.decode("utf-8"),'old_model_predictions': old_model_predictions.decode("utf-8")})
    except:
       return jsonify({'Error': 'Error occur'})


if __name__ == '__main__':
    app.run(debug=True)
