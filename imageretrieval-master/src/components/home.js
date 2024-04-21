import React from 'react';
import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import image1 from '../images/image_1.png';
import image2 from '../images/image_2.png';
import image3 from '../images/image_3.png';
import image4 from '../images/image_4.png';
import image5 from '../images/image_5.png';
import image6 from '../images/image_6.png';
import videoSrc from '../images/video.mp4';

function ImageSlider() {
  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 3000,
  };
  const images = [image1, image2, image3, image4, image5];

  return (
    <div className="image-slider">
      <h1 style={{ fontSize: '2.5rem', textAlign: 'center', color: '#333', fontWeight: 'bold', marginBottom: '20px' }}>Content-Based Image Retrieval</h1>

      <ul className='d-flex justify-content-center' style={{ listStyleType: 'none' }}>
        <a href="https://www.linkedin.com/in/avadhut-jadhav-9b5884258/"><li style={{ display: 'inline-block', margin: '0 5px', textDecoration: 'underline' }}>Avadhut Jadhav</li></a>
        <a href="https://www.linkedin.com/in/preet-savalia-a06b3b258?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"><li style={{ display: 'inline-block', margin: '0 5px', textDecoration: 'underline' }}>Preet Savalia</li></a>
        <a href="https://www.linkedin.com/in/vedant-funde-aba919258"><li style={{ display: 'inline-block', margin: '0 5px', textDecoration: 'underline' }}>Vedant Funde</li></a>
        <a href="https://www.linkedin.com/in/harshit-raj-9bb2911b0/"><li style={{ display: 'inline-block', margin: '0 5px', textDecoration: 'underline' }}>Harshit Raj</li></a>
        <a href="https://www.linkedin.com/in/malhargadge/"><li style={{ display: 'inline-block', margin: '0 5px', textDecoration: 'underline' }}>Malhar Gadge</li></a>
      </ul>

      <h2 className="text-center">| Paper | Code | Dataset | Poster | Slides | Short Talk</h2>
      <Slider {...settings}>
        {images.map((image, index) => (
          <div key={index}>
            <img src={image} alt={`Slide ${index + 1}`} style={{ width: '80%', margin: '0 auto', height: 'auto' }} />
          </div>
        ))}
      </Slider>

      <div className="abstract-container" style={{ textAlign: 'justify', textJustify: 'inter-word', maxWidth: '1000px', margin: '0 auto' }}>
        <h2 className="text-center">Abstract</h2>
        <p className="abstract-text">
          In the realm of computer vision, efficient image retrieval stands as a cornerstone for various applications, ranging from content-based search engines to medical diagnostics. This project, titled "Image Retrieval," endeavors to craft a robust system capable of swiftly locating similar images given an input query. Utilizing relevant techniques, including Convolutional Neural Networks (CNNs) for feature extraction and advanced dimensionality reduction techniques such as Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA), alongside a number of machine learning classifiers, the project delves into the intricate realm of image classification and retrieval. The CIFAR-10 dataset serves as the bedrock for training and validation.
          We begin our project with the acquisition and preprocessing of the CIFAR-10 dataset, splitting images into their constituent red, green, and blue channels. Employing a pre-trained ResNet-50 model, features are extracted, while PCA and LDA are used to reduce the high-dimensional data, distilling crucial insights while mitigating computational overhead.
          A collection of classifiers, including Decision Trees, K-Nearest Neighbors (KNN), Random Forests, Logistic Regression, Support Vector Machines (SVM), and Gaussian Naive Bayes, is applied with the reduced-dimensional feature vectors, each having a different accuracy on the test dataset.
          The deployment of the project lies in the realm of image retrieval, where an SVM classifier with an RBF kernel emerges as the most accurate and able classifier. Predicting the label of the input images, similar images are produced through the use of Nearest Neighbors (NN). These retrieved images serve as a testament to the efficacy and prowess of the retrieval system.
          In conclusion, the project not only highlights the viability of CNNs and machine learning domains in the domain of image retrieval but also lays the cornerstone for further explorations and advancements in the ever-evolving landscape of computer vision and image processing.
        </p>
      </div>

      <h2 className="text-center">Keywords:</h2>
      <p className="text-center">CNN, KNN, SVM, PCA, LDA, Naive Bayes, Decision Trees</p>
      
      {/* Example Classes Section */}
      <div className="example-classes">
        <h2 className="text-center">Example Classes</h2>
        <figure className="text-center">
          <img src={image6} style={{ width: '1000px', height: 'auto', margin: '0 10px' }} alt="Example Classes" />
        </figure>
      </div>
      <div className="execution">
        <h2 className="text-center">Execution</h2>
        <video controls style={{ width: '80%', margin: '0 auto', display: 'block' }}>
          <source src={videoSrc} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>

      <div className="image-slider">
        <h2 className="text-center">References:</h2>
        <ul style={{ display: 'flex', listStyleType: 'none', justifyContent: 'center', alignItems: 'center', margin: '0', padding: '0' }}>
          <li style={{ margin: '0 10px' }}><a href="https://github.com/anandmishra22/PRML-Spring-2023/blob/main/Project/Reference_Codes/refCodes4PRMLProject.ipynb?short_path=b250b2c">Anand Mishra-CNN Reference Code</a></li>
          <li style={{ margin: '0 10px' }}><a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a></li>
          <li style={{ margin: '0 10px' }}><a href="https://arxiv.org/abs/1311.2901">Visualizing and Understanding Convolutional Networks</a></li>
          <li style={{ margin: '0 10px' }}><a href="https://arxiv.org/abs/1201.0490">Scikit-learn: Machine Learning in Python</a></li>
        </ul>
      </div>

      <div className="links text-center">
        <h2>Links:</h2>
        <a href="https://github.com/vedantfunde/supreme-giggle">Code Link, </a>
        <a href="https://www.overleaf.com/8823481134kffwdphgsnsx#497c35">Report Link,</a>
        <a href="https://docs.google.com/presentation/d/1TsOftL0aE4uKKje-xW4Bg0LNwqW71GmgFuyIYiNuKLg/edit?usp=sharing">Slides</a>
      </div>

      <div className="text-center">
        <div>
          <h2>Team</h2>
          {/* Add team members here */}
        </div>
      </div>

      <h2 className="text-center">Acknowledgement:</h2>
      <p className="text-center">This project was conducted as part of the PRML course under the guidance of Professor Anand Mishra.</p>

      <h2 className="text-center">Contact:</h2>
      <p className="text-center">For any inquiries, please contact Avadhut Jadhav at avadhutjadhav966@gmail.com or raise an issue on GitHub at <a href="https://github.com/vedantfunde/supreme-giggle">https://github.com/vedantfunde/supreme-giggle</a>.</p>
    </div>
  );
}

export default ImageSlider;
