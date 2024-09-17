### Parking lot detection (using OpenCV)
- Ferrari Davide
- Frighetto Eddy
- Manuzzato Matteo
Parking lot analysis is an algorithm designed for detecting and classifying parking spaces. It processes each image in a parking lot dataset, identifying and classifying both occupied and unoccupied parking spaces, as well as detecting and classifying cars within and outside the parking areas. The algorithm also visualizes the parking lot's status on a map.

### To run the code:

1. Navigate to the `build` folder:

   ```cd build```
   
3. Execute the CMake configuration:

   ```cmake ..```
   
4. Build the project:
    
    ```make```
   
5. Return to the main folder:
    
    ```cd ..```
   
6. Run the program:

   ```./ParkingLot_analysis```

### To save the result images and metrics, run:

  ```./ParkingLot_analysis --save```

All files will be stored in the `results` folder.
