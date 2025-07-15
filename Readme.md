TITLE : AI-Based Enhancement of Fingerprint Biometrics for Blood Group Recognition


Blood group detection is crucial in medical diagnosis, blood transfusion safety, and emergency medicine. This study discusses a deep learning-based method for discrimination of blood groups based on fingerprint features. This study analyzes several deep learning architectures - Convolutional Neural Networks (CNNs), MobileNet, ResNet with Recurrent Neural Networks (RNNs), Vision Transformer models, in order to evaluate their performance on a publicly released fingerprint-based blood group dataset. The method adopted involves data preprocessing and using techniques of data augmentation to provide the model with robustness and generalization. Feature extraction has been taken from a baseline CNN, whereas MobileNet was focused on computation as being lightweight and efficient. ResNet+RNN used residual learning to integrate sequential patterns of recognition into an architecture to boost the classification accuracy. Lastly, the attention mechanism of Vision Transformer model uses intricate details about fingerprint significantly enhancing the blood group classification. Experimental results show that Vision Transformer outperforms other architectures, achieving state-of-the-art accuracy and indicating that it is indeed focusing on the fingerprint features that matter. The approach proposed here brings about a novel and efficient solution for the identification of blood groups using the latest machine learning techniques. This study contributes to the growing area of data-driven healthcare research and offers a scalable framework for similar classification tasks.



Use python version 
3.10.13
transformers==4.19.0
tensorflow==2.15.0
scikit-learn==1.5.0


Note: For B+ and AB+ , B- and AB- as the images are similar the model is performing less accurately, for the other classes the model is working fine.
