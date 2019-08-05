# W251_AEI_Final_Project
Stolen car detection using ALPR and Car Classification

Leverage two models for identifying cars that can be used in identifying cars from images, video, or live cameras

* ALPR - Automatic License Plate Recongition

  *leveraging https://github.com/sergiomsilva/alpr-unconstrained* 
  
* Car Recognition 

   *leveraging https://github.com/foamliu/Car-Recognition*
   * Evaluate_image.ipynb

- run the “evaluate_image” notebook
- you can change the “victim_no” to your number
- it’s pulling form a sample of 3 images with one of them designed to trigger the stolen car alert (the Acura)
- the capture_image notebook is designed to take pictures from an open webcam and will input into the images file
- let me know if works we can convert to python file that can be run with arguments from terminal
 - list of stolen cars is a live list from that website so will change as it changes

* Jetson TX2 live camera 
  * Demo7_fromCamera.py
