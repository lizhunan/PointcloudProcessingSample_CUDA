#include "display.h"
#include <pangolin/pangolin.h>

namespace downsampling {

Display::Display(const std::string win_name)
{
    thread = std::thread(std::bind(&Display::show, this));
    this->win_name = win_name;
}

void Display::set_points(const float* points, const int points_num)
{   
    this->points = new float[points_num*POINT_DIM];
    memcpy(this->points, points,sizeof(float)*points_num*POINT_DIM);
    this->points_num = points_num;
}

void Display::show()
{
    pangolin::CreateWindowAndBind(win_name, 640, 480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,1000),
        pangolin::ModelViewLookAt(0,-50,10, 0,0,10, pangolin::AxisZ)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
        .SetHandler(&handler);
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        glPointSize(2);

        glBegin(GL_POINTS);
        glColor3f(1,1,1);
        for (int i=0; i<this->points_num; i++)
        {
            glVertex3f(points[i*POINT_DIM+0], points[i*POINT_DIM+1], points[i*POINT_DIM+2]);
        }
        glEnd();

        pangolin::FinishFrame();
    }
    pangolin::DestroyWindow(win_name);
}

Display::~Display()
{
    if (thread.joinable()) thread.join();
}

}