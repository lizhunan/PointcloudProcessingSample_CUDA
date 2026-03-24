#include "display.h"
#include <pangolin/pangolin.h>

namespace display {

Display::Display(const std::string win_name)
{
    thread = std::thread(std::bind(&Display::show, this));
    this->win_name = win_name;
}

void Display::set_pointcloud_xyz(const float* points, const int points_num)
{	
	this->points = new float[points_num*POINT_DIM];
	memcpy(this->points, points,sizeof(float)*points_num*POINT_DIM);
	this->points_num = points_num;
}

void Display::set_neighbors(const int* neighbors, const int k, const int point_idx)
{
	if (this->neighbors != nullptr) {
        delete[] this->neighbors;
        this->neighbors = nullptr;
    }

	this->neighbors = new int[k+1];
	this->neighbors_k = k;
	int base_idx = point_idx * k;
	for (int i=0; i<k+1; i++)
	{
		this->neighbors[i] = neighbors[base_idx + i];
	}
}

void Display::set_normals(const float* normals, const int points_num)
{
	this->normals = new float[points_num*3];
	memcpy(this->normals, normals,sizeof(float)*points_num*3);
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
		// glColor3f(0,1,0);
		for (int i=0; i<points_num; i++)
		{
			for (int j=0; j<this->neighbors_k; j++)
			{
				if (this->neighbors[j] == points[i*POINT_DIM+4]) glColor3f(1,0,0);
			}
			
			glVertex3f(points[i*POINT_DIM+0], points[i*POINT_DIM+1], points[i*POINT_DIM+2]);
			glColor3f(0,1,0);
		}
		glEnd();

		if (normals != nullptr){
			glLineWidth(1.0f);
			glColor3f(0,0,1); // blue normals
			glBegin(GL_LINES);
			for (int i=0; i<points_num; i++)
			{
				float px = points[i*POINT_DIM+0];
				float py = points[i*POINT_DIM+1];
				float pz = points[i*POINT_DIM+2];
				float nx = normals[i*3 + 0];
				float ny = normals[i*3 + 1];
				float nz = normals[i*3 + 2];
				glVertex3f(points[i*POINT_DIM+0], points[i*POINT_DIM+1], points[i*POINT_DIM+2]);

				glVertex3f(px + nx * 10,
							py + ny * 10,
							pz + nz * 10);
			}
			glEnd();
		}
		pangolin::FinishFrame();
    }
    pangolin::DestroyWindow(win_name);
}

Display::~Display()
{
	if (thread.joinable()) thread.join();
}

}