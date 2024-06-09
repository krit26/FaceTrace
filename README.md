# FaceTrace
Face Recognition Project

### Things to know:
- `<base_path>`: persisted directory which will maintain image dataset as well as the model weights 
- `<base_path>/images/<user-id>/image.jpeg`: conventional path an image in dataset
- `<base_path>/metadata.json`: metadata json contain all the precomputed image vectors and maintains all the metadata of an image in dataset
- `<base_path>/weights/<model-name>/weights.h5`: this folder will contain the weights of the facenet models.
  - copy the `/weights` folder from repo to the `<base_path>`

### API Contract
Face Recognition API support following endpoints:
1. `/face-detect`: detects the bounding box of facial area in the input image
2. `/represent`: detects the facial areas input image and its corresponding embeddings
3. `/verity`: given 2 images in inputs, it verifies if both are matching or not.
4. `/add`: api maintain an image store, in which client can add new images which will be used later for recognition purpose.
5. `/recognize`: for given input, it will try to verify it with existing image in image store

Following are the request response payload for each endpoint

- Request

    - `/face-detect`, `/represent`, `/recognize`
        ```
        {
          "payloads": [
            {
              "image": "<base64_encoded_image_string>"
            },
            {
              "image": "<base64_encoded_image_string>"
            }
          ]
        }
        ```
    - `/verify`
        ```
        {
          "payloads": [
            {
              "image1": "<base64_encoded_image_string>",
              "image2": "<base64_encoded_image_string>",
            },
            {
              "image1": "<base64_encoded_image_string>",
              "image2": "<base64_encoded_image_string>",
            }
          ]
        }
        ```
    - `/add`
       ```
        {
          "payloads": [
            {
              "image": "<base64_encoded_image_string>",
              "userId": "<Unique id of the user>"
            },
            {
              "image": "<base64_encoded_image_string>",
              "userId": "<Unique id of the user>"
            }
          ]
        }
        ```
      

- Response

    - `/face-detect`, `/represent`
        ```
        {
          "results": [
            {
              "faces": [
                {
                  "x": 0,
                  "y": 0,
                  "w": 250,
                  "h": 250,
                  "confidence": 0.99,
                  "left_eye": (25, 125),
                  "right_eye": (100, 145),
                  "embedding": [0.2, 0.4, 0.5] # only when endpoint is /represent
                }
              ]
            },
            {
              "faces": [
                {
                  "x": 0,
                  "y": 0,
                  "w": 250,
                  "h": 250,
                  "conf": 0.99,
                  "left_eye": (25, 125),
                  "right_eye": (100, 145),
                  "embedding": [0.2, 0.4, 0.5] # only when endpoint is /represent
                }
              ]
            }
          ]
        }
        ```
    - `/add`
        ```
        {
          "results": [
            {
              "success": True
            },
            {
              "success": False,
              "reason": "trying to add duplicate image. image file "
            }
          ]
       }
       ```
    - `/verify`
       ```
       {
         "results": [
           {
             "verified": true,
             "distance": 0.83,
             "metric": "cosine_similarity",
             "threshold": 0.6,
             "embedding_model": "FaceNet512",
             "detector_model": "FastMtcnn",
             "faces": {
               "image1": {
                 "x": 0,
                 "y": 0,
                 "w": 250,
                 "h": 250,
                 "confidence": 0.99,
                 "left_eye": (25, 125),
                 "right_eye": (100, 145),
               },
               "image2": {
                 "x": 0,
                 "y": 0,
                 "w": 250,
                 "h": 250,
                 "conf": 0.99,
                 "left_eye": (25, 125),
                 "right_eye": (100, 145),
               }
             }
           }
         ]
      }
      ```
    - `/recognize`
      ```
      {
        "results": [
          {
            "verified": true,
            "distance": 0.83,
            "metric": "cosine_similarity",
            "threshold": 0.6,
            "embedding_model": "FaceNet512",
            "detector_mode": "FastMtcnn",
            "userId": "<Unique Id of the user of matched>"
          }
        ]
      }
      ```