from lib.utility import get_input_directories

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
# Introduced a function to retrieve the directories
#    array_of_exr_dirs = [
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-19-25/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-27-05/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-23--19-00-04/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-19-47/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-27-17/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-23--19-00-32/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-20-05/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-27-34/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-23--19-01-08/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-20-20/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-27-50/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-22-03/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-20-37/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-28-06/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-22-43/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-20-51/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-28-17/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-23-21/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-21-02/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-28-36/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-23-47/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-21-14/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-30-15/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-24-16/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-21-25/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-30-26/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-25-36/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-21-36/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-30-35/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-26-01/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-23-09/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-30-45/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-26-24/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-23-20/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-30-52/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-26-53/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-23-29/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-31-05/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-27-20/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-23-37/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-31-17/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-29-39/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-23-53/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-31-27/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-30-17/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-24-04/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-32-26/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-31-08/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-24-15/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-32-38/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-32-04/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-24-25/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-32-59/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-33-56/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-24-40/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-33-11/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-34-25/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-25-51/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-35-21/EXR_RGBD/depth/", "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-35-15/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-26-15/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-35-30/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-36-15/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-26-35/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-23--18-47-34/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-36-44/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-26-44/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-23--18-53-26/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-05-29--14-37-08/EXR_RGBD/depth/",
#        "/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-26-54/EXR_RGBD/depth/","/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-23--18-54-01/EXR_RGBD/depth/",
#    ]
    array_of_exr_dirs = get_input_directories()
