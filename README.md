# Reverse-Engineering-Model

TL;DR - STL files are used for renders and 3d printing, and many times people publish stl files online, but they can't be easily modified without manually reverse engineering it.
This tool would be able to intelligently produce a parameterized 3d-model from a static 3d-model.

---HOW TO USE----

git clone https://github.com/PlantTheDiffuser/REM-1.git
source venv/bin/activate
pip install -r requirements.txt

-----------------


The way that STL, or stereolithography file works, is a file that stores a series of 3 coordinate points that formes many triangles that can be rendered into a 3d shape. The way that CAD software works however, is by using a set of sketches on a 2d plane, and applying features to that sketch such as extrudes, revolves, etc. Instead of a list of points, it stores the file as a set of instructions to be rendered(example image below).

You can easily convert those CAD models into stl files that approximate curves and surfaces with a series of triangles, but isn't quite perfect. Thats why you can't go back from an STL file to a CAD model without any kind of intelligence. You can imagine the problem like converting a PDF of text into a png. You can easily take a screenshot of a document, but it is difficult to read an image and convert it into text. I want to do the same thing, but for 3d space. I haven't found any software that does this easily.

So far, I'm doing this by converting the 3d stl model into a point cloud, and using that as the input layer for a neural network. The first issue I'm running into is running out of memory space. To get an accurate reading, I need to read thousands of points, and do many calculations on them all. If you run the TrainTest python notebook, it will likely slow down your whole computer before crashing. And that's with only 5 files of training data and 1 hidden layer. I need a way to optimize the operations by discarding irrelevant information. There is still a lot to learn about NN, and tricks to getting past these issues.

![image](https://github.com/PlantTheDiffuser/Reverse-Engineering-Model/assets/59662694/b4835276-f27e-4ff4-b682-9449b6e9380b)
This shape is made by drawing a sketch and revolving it around the line in the center. It is a smooth shape defined not as a list of coordinates, but as an equation.

![image](https://github.com/PlantTheDiffuser/Reverse-Engineering-Model/assets/59662694/d5b55999-db1e-45ca-994d-7ce27535d481)
The same shape, saved as an STL file looks like a set of triangles creating a surface. No software can read this as a circle.

TL;DR - STL files are used for renders and 3d printing, and many times people publish stl files online, but they can't be easily modified without manually reverse engineering it.
This tool would be able to intelligently produce a parameterized 3d-model from a static 3d-model.
