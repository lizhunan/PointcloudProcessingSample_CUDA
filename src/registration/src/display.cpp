#include "display.h"
#include <pangolin/pangolin.h>

namespace reg_display {

Display::Display(const std::string win_name)
{
    thread = std::thread(std::bind(&Display::show, this));
    this->win_name = win_name;
}

void Display::set_source(const float* points, const int points_num)
{   
    this->source = new float[points_num*POINT_DIM];
    memcpy(this->source, points,sizeof(float)*points_num*POINT_DIM);
    this->source_num = points_num;
}

void Display::set_target(const float* points, const int points_num)
{   
    this->target = new float[points_num*POINT_DIM];
    memcpy(this->target, points,sizeof(float)*points_num*POINT_DIM);
    this->target_num = points_num;
}

void Display::set_corr(const float* source, const float* target, const int corr_num)
{
    this->source_corr = new float[corr_num*POINT_DIM];
    this->target_corr = new float[corr_num*POINT_DIM];
    memcpy(this->source_corr, source, sizeof(float)*corr_num*POINT_DIM);
    memcpy(this->target_corr, target, sizeof(float)*corr_num*POINT_DIM);
    this->corr_num = corr_num;
}


void Display::set_feat(const float* src, const float* tar, const float* src_feat, const float* tar_feat, 
                  	   const int src_feat_num, const int tar_feat_num)
{	
	this->source = new float[src_feat_num*POINT_DIM];
	this->target = new float[tar_feat_num*POINT_DIM];
	this->src_feat = new float[src_feat_num*33];
    this->tar_feat = new float[tar_feat_num*33];
    memcpy(this->source, src,sizeof(float)*src_feat_num*POINT_DIM);
    memcpy(this->target, tar,sizeof(float)*tar_feat_num*POINT_DIM);
	memcpy(this->src_feat, src_feat,sizeof(float)*src_feat_num*33);
    memcpy(this->tar_feat, tar_feat,sizeof(float)*tar_feat_num*33);
    this->source_num = src_feat_num;
    this->target_num = tar_feat_num;
}

void Display::set_feat_match(const float* src_matched, const float* tar_matched, const int matched_num)
{
    this->src_matched = new float[matched_num*POINT_DIM];
    this->tar_matched = new float[matched_num*POINT_DIM];
    memcpy(this->src_matched, src_matched, sizeof(float)*matched_num*POINT_DIM);
    memcpy(this->tar_matched, tar_matched, sizeof(float)*matched_num*POINT_DIM);
    this->matched_num = matched_num;
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

        // target
        glBegin(GL_POINTS);
        glColor3f(1,1,1);
        for (int i=0; i<target_num; i++)
        {
			if (target[i*POINT_DIM+5] == 0)
			{
                // for (int j = 0; j < 33; j++)
                // {
                //     printf("target feat[%d] = %.6f\n", j, tar_feat[i * 33 + j]);
                // }
                // printf("\n");
				glColor3f(0,1,0);
			}
			else glColor3f(1,1,1);
            glVertex3f(target[i*POINT_DIM+0], target[i*POINT_DIM+1], target[i*POINT_DIM+2]);
        }
        glEnd();

        // source
        glBegin(GL_POINTS);
        glColor3f(1,0,0);
        for (int i=0; i<source_num; i++)
        {
			if (source[i*POINT_DIM+5] == 0) 
			{
                // for (int j = 0; j < 33; j++)
                // {
                //     printf("source feat[%d] = %.6f\n", j, src_feat[i * 33 + j]);
                // }
                // printf("\n");
				glColor3f(0,1,0);
			}
			else glColor3f(1,0,0);
            glVertex3f(source[i*POINT_DIM+0], source[i*POINT_DIM+1], source[i*POINT_DIM+2]);
        }
        glEnd();

		// feat matched line
        for (int i=0; i<matched_num; i++)
        {
			// printf("i=%d, x_s: %f, y_s: %f, z_s: %f, x_t: %f, y_t: %f, z_t: %f\n", 
			// 		i, src_matched[i*POINT_DIM+0], src_matched[i*POINT_DIM+1], src_matched[i*POINT_DIM+2],
			// 		tar_matched[i*POINT_DIM+0], tar_matched[i*POINT_DIM+1], tar_matched[i*POINT_DIM+2]);
        	glLineWidth(1);
        	glBegin(GL_LINES);
        	glColor3f(0,1,0);
        	glVertex3f(src_matched[i*POINT_DIM+0], src_matched[i*POINT_DIM+1], src_matched[i*POINT_DIM+2]);
        	glVertex3f(tar_matched[i*POINT_DIM+0], tar_matched[i*POINT_DIM+1], tar_matched[i*POINT_DIM+2]);
        	glEnd();
        }

        // correspondence line
        for (int i=0; i<corr_num; i++)
        {
         glLineWidth(1);
         glBegin(GL_LINES);
         glColor3f(0,1,0);
         glVertex3f(source_corr[i*POINT_DIM+0], source_corr[i*POINT_DIM+1], source_corr[i*POINT_DIM+2]);
         glVertex3f(target_corr[i*POINT_DIM+0], target_corr[i*POINT_DIM+1], target_corr[i*POINT_DIM+2]);
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