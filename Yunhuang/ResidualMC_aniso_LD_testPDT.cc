//#define X 1.0               		//1D Problem Domain -- Right Boundary
#define M_PI 3.14159265358   		  //define Pi
#define Nparticles 1e5      		  //Number of particles per batch
#define Nbatches 10								//Number of batches

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <assert.h>

/*=========== Sample scattering angle in Lab frame =============*/
void scat_angle_lab(double mu_in, const std::vector<double>& mu_d, const std::vector<double>& wt_d, double& mu_out)
{
  unsigned int num_d_angles = mu_d.size();
  assert(num_d_angles == wt_d.size());  //check to see if sizes of mu_d and wt_d match
  
  std::vector<double> CWF(num_d_angles, 0.0); //cumulative weight function
  double S = 0.0;    //sum of weight
  for(unsigned int i=0; i<num_d_angles; i++)
  {
    S += wt_d[i];
    CWF[i] = S;
  }
  assert( std::fabs(S - 1.0) < 1e-6);
  
  unsigned int i_d;   //sampled discrete scattering interval index
  double s;           //random number
  s = float(rand())/float(RAND_MAX);
  //sample from the discrete scattering angle distribution in CoM frame
  for(unsigned int i=0; i<num_d_angles; i++)
  {
    if(s*S <= CWF[i])  //found the encompassing weight interval
    {
      i_d = i;
      break;
    }
  }
  //sample azimuthal angle
  s = float(rand())/float(RAND_MAX);
  double w_s = 2.0*M_PI*s;
  //compute the out-going scattering angle in Lab-frame
  mu_out=mu_in*mu_d[i_d]-sqrt((1-mu_in*mu_in)*(1-mu_d[i_d]*mu_d[i_d]))*cos(w_s);
  
}

/*============ Sample distance to next collision ==============*/
void ds_btw_collision(double sigma_t, double& ds)
{
  //sample distance to next collision
  // ds is to next collision
  double s = float(rand())/float(RAND_MAX);
  ds = -log(1.0-s)/sigma_t;
}

/*============ Find Cell after Collision ==============*/
int FindCell(const std::vector<double>& x_b, double x_new)
{
	int num_cells = x_b.size() - 1;
	int cell_id = -1;
	for(int i=0;i<num_cells;i++)
	{
		if((x_b[i]<=x_new)&&(x_b[i+1]>x_new))  //determine current cell the particle is in
		{
			cell_id=i; 
			break;
		}
		if(x_new==x_b[num_cells])
			cell_id=num_cells-1;
		if(x_new > x_b[num_cells])
		  cell_id = num_cells;
	}
	return(cell_id);
}

/*============ Particle Ray Tracing ==============*/
void RayTracing(const std::vector<double>& x_b, double x, double ds, double mu, 
                int i, unsigned int num_cells,
                const std::vector<double>& mu_d, const std::vector<double>& wt_d,
                const std::vector<double>& sig_t, const std::vector<double>& sig_s,
                double weight, std::vector<double>& d_phi)
{
  double x_new, mu_out, s;
  while(1)
  {
    x_new = x + ds*mu;
    if( i==0 && x_new < x_b[0] ) //leak through left boundary
    {
      d_phi[i] += std::fabs(x/mu)*weight;
      break;
    }
    else if( i==int(num_cells)-1 && x_new > x_b[num_cells] ) //leak through right boundary
    {
      d_phi[i] += std::fabs((x_b[num_cells] - x)/mu)*weight;
      break;
    }
    else if( i < FindCell(x_b, x_new) )
    {
      d_phi[i] += std::fabs((x_b[i+1] - x)/mu)*weight;
      x=x_b[i+1];		//bring particle to boundary
      if( i==int(num_cells)-1 )
        break;
      else
        i++;
      ds_btw_collision(sig_t[i], ds);		//sample distance between collision
    }
    else if( i > FindCell(x_b, x_new) )
    {
      d_phi[i] += std::fabs((x -  x_b[i])/mu)*weight;
      x=x_b[i];		//bring particle to boundary
      if( i==0 )
        break;
      else
        i--;
      ds_btw_collision(sig_t[i], ds);		//sample distance between collision
    }
    else  //Collision happened in the same cell
    /*=============== resample post-scattering angle ========================*/
    {
      d_phi[i] += ds*weight;
      s = float(rand())/float(RAND_MAX);
      if(s <= (sig_t[i] - sig_s[i])/sig_t[i])		//if absorbed, terminate current history
        break;
      else
      {
        x=x_new;				//confirm x postion after collision
        scat_angle_lab(mu, mu_d, wt_d, mu_out);  //sample post-scatter mu in lab frame
        mu = mu_out;
        ds_btw_collision(sig_t[i], ds);		//sample distance between collision
      }
    }
  }
}

/*============ Main Function ==============*/
int main()
{
  int num_quads = 2; //example of S4
  int num_cells = 20;
  int num_dofs = 2*num_cells;  //num_dof
  
  std::vector<double> mu_d;  //discrete scattering mu
  std::vector<double> wt_d;  //discrete scattering wt
  
  //assume isotropic discrete scattering with P3 expansion
  mu_d.push_back(-0.5773502692);
  mu_d.push_back(0.5773502692);
  wt_d.push_back(0.5000000000);
  wt_d.push_back(0.5000000000);
  
  std::vector<double> mu_m(num_quads, 0.0);  //Sn quadrature points (mu's)
  std::vector<double> wt_m(num_quads, 0.0);  //Sn quadrature weights
  std::vector< std::vector<double> > mu_b(num_quads, std::vector<double>(2, 0.0)); //mu boundaries
  std::vector<double> d_mu(num_quads, 2.0/num_quads); //mu interval width
  
  std::vector<double> dx(num_cells, 0.5);    //cell width
  std::vector<double> x_b(num_cells+1, 0.0);  //cell boundaries
  
  //distributed source q, indexing (i_m, i, i_boundary)
  std::vector< std::vector< std::vector<double> > > q(num_quads, 
                                                      std::vector< std::vector<double> >(num_cells,
                                                      std::vector<double>(2, 0.0)));
  //boundary condition (incident angular flux)
  double psi_L = 20.0;
  double psi_R = 0.0;
  
  std::vector<double> sig_t(num_cells, 4.9435);  //Sigma_t
  std::vector<double> sig_s(num_cells, 4.9400);  //Sigma_s
  
  //angular flux container, indexing (i_m, i, i_boundary)
  std::vector< std::vector< std::vector<double> > > psi(num_quads, 
                                                        std::vector< std::vector<double> >(num_cells,
                                                        std::vector<double>(2, 0.0)));
                                                        
  //scalar flux container, indexing (i, i_boundary)
  std::vector< std::vector<double> > phi(num_cells, std::vector<double>(2, 0.0));
      

  /*============ Read in quadrature, angular flux, and source ============*/
  //hard-coded S2 gauss quadrature
  mu_m[0] = -0.70710678118;  wt_m[0] = 1.0;
  mu_m[1] =  0.70710678118;  wt_m[1] = 1.0;

  // //hard-coded S4 gauss quadrature
  // mu_m[0] = -0.861136;  wt_m[0] = 0.347855;
  // mu_m[1] = -0.339981;  wt_m[1] = 0.652145;
  // mu_m[2] =  0.339981;  wt_m[2] = 0.652145;
  // mu_m[3] =  0.861136;  wt_m[3] = 0.347855;

  // //hard-coded S8 gauss quadrature
  // mu_m[0] = -0.9602898564975363;  wt_m[0] = 0.1012285362903763;
  // mu_m[1] = -0.7966664774136267;  wt_m[1] = 0.2223810344533745;
  // mu_m[2] = -0.5255324099163290;  wt_m[2] = 0.3137066458778873;
  // mu_m[3] = -0.1834346424956498;  wt_m[3] = 0.3626837833783620;
  // mu_m[4] =  0.1834346424956498;  wt_m[4] = 0.3626837833783620;
  // mu_m[5] =  0.5255324099163290;  wt_m[5] = 0.3137066458778873;
  // mu_m[6] =  0.7966664774136267;  wt_m[6] = 0.2223810344533745;
  // mu_m[7] =  0.9602898564975363;  wt_m[7] = 0.1012285362903763;
  
  // sum dummy angular flux (testing with isotropic, cell constant values)
  // psi = 2 in cell #0 - #4, psi_m = 1 in cell# 6-9, psi_m linear in cell #5
  psi[0][0][0] = 1.507334233932562;
  psi[0][0][1] = 1.338775662063302;
  psi[0][1][0] = 1.34154022813126;
  psi[0][1][1] = 1.19067857639093;
  psi[0][2][0] = 1.193137322931606;
  psi[0][2][1] = 1.058016893434433;
  psi[0][3][0] = 1.060201694467467;
  psi[0][3][1] = 0.9390708510477155;
  psi[0][4][0] = 0.9410100292855973;
  psi[0][4][1] = 0.8322984900691099;
  psi[0][5][0] = 0.8340171840800317;
  psi[0][5][1] = 0.7363156650508793;
  psi[0][6][0] = 0.7378361551440066;
  psi[0][6][1] = 0.6498781008158883;
  psi[0][7][0] = 0.6512200978736168;
  psi[0][7][1] = 0.5718652622317626;
  psi[0][8][0] = 0.5730461632332328;
  psi[0][8][1] = 0.5012658280984762;
  psi[0][9][0] = 0.5023009416468076;
  psi[0][9][1] = 0.4371645808401547;
  psi[0][10][0] = 0.4380673256170111;
  psi[0][10][1] = 0.3787305420455616;
  psi[0][11][0] = 0.3795126207656911;
  psi[0][11][1] = 0.3252062000521755;
  psi[0][12][0] = 0.3258777511729681;
  psi[0][12][1] = 0.275897689925353;
  psi[0][13][0] = 0.2764674190780816;
  psi[0][13][1] = 0.2301657985303314;
  psi[0][14][0] = 0.230641091376865;
  psi[0][14][1] = 0.1874176780907679;
  psi[0][15][0] = 0.1878046960687551;
  psi[0][15][1] = 0.147099160811867;
  psi[0][16][0] = 0.1474029210095625;
  psi[0][16][1] = 0.1086875749379247;
  psi[0][17][0] = 0.1089120151341012;
  psi[0][17][1] = 0.07168496911435897;
  psi[0][18][0] = 0.07183299882644296;
  psi[0][18][1] = 0.03561165721747271;
  psi[0][19][0] = 0.03568519542010513;
  psi[0][19][1] = -2.158212144571396e-11;

  psi[1][0][0] = 1.58826962500603;
  psi[1][0][1] = 1.416887392498492;
  psi[1][1][0] = 1.413967555948609;
  psi[1][1][1] = 1.260593250100723;
  psi[1][2][0] = 1.257995494567261;
  psi[1][2][1] = 1.120640850185021;
  psi[1][3][0] = 1.11833149988529;
  psi[1][3][1] = 0.9952159176409392;
  psi[1][4][0] = 0.9931650352683401;
  psi[1][4][1] = 0.8826925040908884;
  psi[1][5][0] = 0.8808735029948641;
  psi[1][5][1] = 0.7816119102228064;
  psi[1][6][0] = 0.780001209757993;
  psi[1][6][1] = 0.6906637758733052;
  psi[1][7][0] = 0.6892404957105357;
  psi[1][7][1] = 0.6086690931050384;
  psi[1][8][0] = 0.6074147825477206;
  psi[1][8][1] = 0.5345649220777269;
  psi[1][9][0] = 0.533463320874741;
  psi[1][9][1] = 0.4673906115773337;
  psi[1][10][0] = 0.4664274391317735;
  psi[1][10][1] = 0.4062753455732182;
  psi[1][11][0] = 0.4054381158143972;
  psi[1][11][1] = 0.3504268543629485;
  psi[1][12][0] = 0.3497047138817838;
  psi[1][12][1] = 0.2991211439614581;
  psi[1][13][0] = 0.2985047313109191;
  psi[1][13][1] = 0.2516931105911364;
  psi[1][14][0] = 0.2511744349278218;
  psi[1][14][1] = 0.2075279186033033;
  psi[1][15][0] = 0.2071002560969347;
  psi[1][15][1] = 0.1660530300581449;
  psi[1][16][0] = 0.165710836726102;
  psi[1][16][1] = 0.1267307826378369;
  psi[1][17][0] = 0.1264696224747832;
  psi[1][17][1] = 0.08905141967578702;
  psi[1][18][0] = 0.08886790714670931;
  psi[1][18][1] = 0.05252648194583189;
  psi[1][19][0] = 0.05241823809701779;
  psi[1][19][1] = 0.01668247554481255;

  for(unsigned int m=0; m<num_quads; m++)
    for(unsigned int i=0; i<num_cells; i++)
    {
      psi[m][i][0] = psi[m][i][0]*4.0*M_PI;
      psi[m][i][1] = psi[m][i][1]*4.0*M_PI;
    }

  
  // Manufactured Solution Source
  // psi_m = 3 in cell #0 - #4, psi_m = 1 in cell# 6-9, psi_m linear in cell #5
  //// Left
  // std::vector<double> psi_m(2, 0.0);
  // std::vector<double> phi_m(2, 0.0);
  // for(unsigned int m=0; m<num_quads; m++)
  // {
  //   for(unsigned int i=0; i<=9; i++)
  //   {
  //     psi_m[0] = 3.0;
  //     psi_m[1] = 3.0;
  //     phi_m[0] = 0.0;  //reset phi
  //     phi_m[1] = 0.0;  //reset phi
  //     for(unsigned int mm=0; mm<num_quads; mm++)
  //     {
  //       phi_m[0] += psi_m[0]*wt_m[mm];
  //       phi_m[1] += psi_m[1]*wt_m[mm];
  //     }
  //     q[m][i][0] = sig_t[i]*psi_m[0] - sig_s[i]/2.0*phi_m[0];
  //     q[m][i][1] = sig_t[i]*psi_m[1] - sig_s[i]/2.0*phi_m[1];
  //   }
  //   //// Middle
  //   psi_m[0] = 3.0;
  //   psi_m[1] = 2.0;
  //   double dpsi_dx = (psi_m[1] - psi_m[0])/dx[10];
  //   phi_m[0] = 0.0;  //reset phi
  //   phi_m[1] = 0.0;  //reset phi
  //   for(unsigned int mm=0; mm<num_quads; mm++)
  //   {
  //     phi_m[0] += psi_m[0]*wt_m[mm];
  //     phi_m[1] += psi_m[1]*wt_m[mm];
  //   }
  //   q[m][10][0] = mu_m[m]*dpsi_dx + sig_t[10]*psi_m[0] - sig_s[10]/2.0*phi_m[0];
  //   q[m][10][1] = mu_m[m]*dpsi_dx + sig_t[10]*psi_m[1] - sig_s[10]/2.0*phi_m[1];
  //   //// Middle
  //   psi_m[0] = 2.0;
  //   psi_m[1] = 1.0;
  //   dpsi_dx = (psi_m[1] - psi_m[0])/dx[11];
  //   phi_m[0] = 0.0;  //reset phi
  //   phi_m[1] = 0.0;  //reset phi
  //   for(unsigned int mm=0; mm<num_quads; mm++)
  //   {
  //     phi_m[0] += psi_m[0]*wt_m[mm];
  //     phi_m[1] += psi_m[1]*wt_m[mm];
  //   }
  //   q[m][11][0] = mu_m[m]*dpsi_dx + sig_t[11]*psi_m[0] - sig_s[11]/2.0*phi_m[0];
  //   q[m][11][1] = mu_m[m]*dpsi_dx + sig_t[11]*psi_m[1] - sig_s[11]/2.0*phi_m[1];
  //   //// Right
  //   for(unsigned int i=12; i<=19; i++)
  //   {
  //     psi_m[0] = 1.0;
  //     psi_m[1] = 1.0;
  //     phi_m[0] = 0.0;  //reset phi
  //     phi_m[1] = 0.0;  //reset phi
  //     for(unsigned int mm=0; mm<num_quads; mm++)
  //     {
  //       phi_m[0] += psi_m[0]*wt_m[mm];
  //       phi_m[1] += psi_m[1]*wt_m[mm];
  //     }
  //     q[m][i][0] = sig_t[i]*psi_m[0] - sig_s[i]/2.0*phi_m[0];
  //     q[m][i][1] = sig_t[i]*psi_m[1] - sig_s[i]/2.0*phi_m[1];
  //   }
  // }
  
  
  
  // build x grid
  for(unsigned int i=0; i<num_cells; i++)
    x_b[i+1] = x_b[i] + dx[i];
    
  //============ Build mu boundaries and intervals ==============//
  for(unsigned int m=0; m<num_quads; m++){
    if( m == 0 ) //first mu interval, set left boundary to -1.0
      mu_b[m][0] = -1.0;
    else
      mu_b[m][0] = (mu_m[m-1] + mu_m[m])/2.0;
      
    if( m == num_quads-1 ) //last mu interval, set right boundary to 1.0
      mu_b[m][1] = 1.0;
    else
    mu_b[m][1] = (mu_m[m] + mu_m[m+1])/2.0;
  }
  
  for(unsigned int m=0; m<num_quads; m++)
    d_mu[m] = mu_b[m][1] - mu_b[m][0];
  //
  //============ Build scalar flux ============//
  for(unsigned int i=0; i<num_cells; i++)
    for(unsigned int m=0; m<num_quads; m++)
    {
      phi[i][0] += psi[m][i][0]*wt_m[m];
      phi[i][1] += psi[m][i][1]*wt_m[m];
    }
  /*============ END: Read in quadrature and angular flux ============*/
  
  //residual in cell interior
  std::vector< std::vector<double> > R_i(num_quads,  std::vector<double>(num_cells, 0.0));  
  
  //residual on cell surface
  //assuming residual is zero on boundaries    
  std::vector< std::vector<double> > R_s(num_quads,  std::vector<double>(num_cells+1, 0.0));
  
  /*============ Build cell-interior residuals ============*/
  for(unsigned int m=0; m<num_quads; m++)
    for(unsigned int i=0; i<num_cells; i++)
    {
      // determine the coefficient in R_i(m,i) expression: R_i = a * mu + b * x + c
      double a, b, c, mu_0, mu_1, R_minus, R_plus, R_top, R_bottom;
      a = - (psi[m][i][1] - psi[m][i][0]) / dx[i];
      b = (- sig_t[i] * (psi[m][i][1] - psi[m][i][0]) + sig_s[i]/2.0 * (phi[i][1] - phi[i][0])
             //weight sums to 2.0 in 1D, 4*PI in 3D
          + (q[m][i][1] - q[m][i][0]))/dx[i];
      c = - sig_t[i] * (psi[m][i][0] - (psi[m][i][1] - psi[m][i][0])*x_b[i]/dx[i])
          + sig_s[i]/2.0 * (phi[i][0] - (phi[i][1] - phi[i][0])*x_b[i]/dx[i])
             //weight sums to 2.0 in 1D, 4*PI in 3D
          + (q[m][i][0] - (q[m][i][1] - q[m][i][0])*x_b[i]/dx[i]);
      
      mu_0 = - (b*x_b[i+1] + c)/a;
      mu_1 = - (b*x_b[i] + c)/a;
      
      // Case 1: r_i(mu, x) = 0 line doesn't cross the rectangular (mu,x) domain,
      //  that is, r_i != 0 in the (mu, x) domain.
      if( (mu_0 >= mu_b[m][1] && mu_1 >= mu_b[m][1])
          || (mu_0 <= mu_b[m][0] && mu_1 <= mu_b[m][0]) )
      {
        R_i[m][i] = std::fabs(a * dx[i] / 2.0 * (std::pow(mu_b[m][1],2) - std::pow(mu_b[m][0],2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * d_mu[m]
                    + c * dx[i] * d_mu[m]);
      }
      // Case 2: r_i has both + and - parts over the entire mu domain
      if( (mu_0 >= mu_b[m][1] && mu_1 <= mu_b[m][0])
          || (mu_0 <= mu_b[m][0] && mu_1 >= mu_b[m][1]) )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_b[m][1], 3) - std::pow(mu_b[m][0], 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_b[m][1], 2) - std::pow(mu_b[m][0], 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_b[m][1] - mu_b[m][0]);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_b[m][1], 3) - std::pow(mu_b[m][0], 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_b[m][1], 2) - std::pow(mu_b[m][0], 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_b[m][1] - mu_b[m][0]);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus);
      }
      // Case 3: 
      if( (mu_b[m][0] < mu_0 && mu_0 < mu_b[m][1]) && mu_1 < mu_b[m][0] )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_0, 3) - std::pow(mu_b[m][0], 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_0, 2) - std::pow(mu_b[m][0], 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_0 - mu_b[m][0]);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_0, 3) - std::pow(mu_b[m][0], 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_0, 2) - std::pow(mu_b[m][0], 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_0 - mu_b[m][0]);
        R_top =   a * dx[i] / 2.0 * (std::pow(mu_b[m][1],2) - std::pow(mu_0,2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_b[m][1] - mu_0)
                    + c * dx[i] * (mu_b[m][1] - mu_0);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus) + std::fabs(R_top);
      }
      // Case 4:
      if( (mu_b[m][0] < mu_0 && mu_0 < mu_b[m][1]) && mu_1 > mu_b[m][1] )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_b[m][1], 3) - std::pow(mu_0, 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_b[m][1], 2) - std::pow(mu_0, 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_b[m][1] - mu_0);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_b[m][1], 3) - std::pow(mu_0, 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_b[m][1], 2) - std::pow(mu_0, 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_b[m][1] - mu_0);
        R_bottom =   a * dx[i] / 2.0 * (std::pow(mu_0,2) - std::pow(mu_b[m][0],2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_0 - mu_b[m][0])
                    + c * dx[i] * (mu_0 - mu_b[m][0]);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus) + std::fabs(R_bottom);
      }
      // Case 5:
      if( (mu_b[m][0] < mu_1 && mu_1 < mu_b[m][1]) && mu_0 < mu_b[m][0] )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_1, 3) - std::pow(mu_b[m][0], 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_1, 2) - std::pow(mu_b[m][0], 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_1 - mu_b[m][0]);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_1, 3) - std::pow(mu_b[m][0], 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_1, 2) - std::pow(mu_b[m][0], 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_1 - mu_b[m][0]);
        R_top =   a * dx[i] / 2.0 * (std::pow(mu_b[m][1],2) - std::pow(mu_1,2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_b[m][1] - mu_1)
                    + c * dx[i] * (mu_b[m][1] - mu_1);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus) + std::fabs(R_top);
      }
      // Case 6:
      if( (mu_b[m][0] < mu_1 && mu_1 < mu_b[m][1]) && mu_0 > mu_b[m][1] )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_b[m][1], 3) - std::pow(mu_1, 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_b[m][1], 2) - std::pow(mu_1, 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_b[m][1] - mu_1);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_b[m][1], 3) - std::pow(mu_1, 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_b[m][1], 2) - std::pow(mu_1, 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_b[m][1] - mu_1);
        R_bottom =   a * dx[i] / 2.0 * (std::pow(mu_1,2) -  std::pow(mu_b[m][0],2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_1 - mu_b[m][0])
                    + c * dx[i] * (mu_1 - mu_b[m][0]);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus) + std::fabs(R_bottom);
      }
      // Case 7:
      if( (mu_b[m][0] < mu_0 && mu_0 < mu_b[m][1]) && (mu_b[m][0] < mu_1 && mu_1 < mu_b[m][1]) 
          && (mu_0 > mu_1) )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_0, 3) - std::pow(mu_1, 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_0, 2) - std::pow(mu_1, 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_0 - mu_1);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_0, 3) - std::pow(mu_1, 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_0, 2) - std::pow(mu_1, 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_0 - mu_1);
        R_top =   a * dx[i] / 2.0 * (std::pow(mu_b[m][1],2) - std::pow(mu_0,2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_b[m][1] - mu_0)
                    + c * dx[i] * (mu_b[m][1] - mu_0);
        R_bottom =   a * dx[i] / 2.0 * (std::pow(mu_1,2) -  std::pow(mu_b[m][0],2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_1 - mu_b[m][0])
                    + c * dx[i] * (mu_1 - mu_b[m][0]);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus) + std::fabs(R_top) + std::fabs(R_bottom);
        
      }
      // Case 8:
      if( (mu_b[m][0] < mu_0 && mu_0 < mu_b[m][1]) && (mu_b[m][0] < mu_1 && mu_1 < mu_b[m][1]) 
          && (mu_0 < mu_1) )
      {
        R_minus = - a*a/(6.0*b) * (std::pow(mu_1, 3) - std::pow(mu_0, 3))
                   - a/2.0*(x_b[i] + c/b) * (std::pow(mu_1, 2) - std::pow(mu_0, 2))
                   - (c*c/(2.0*b) + b/2.0*x_b[i]*x_b[i] + c*x_b[i]) * (mu_1 - mu_0);
        R_plus =  a*a/(6.0*b) * (std::pow(mu_1, 3) - std::pow(mu_0, 3))
                   + a/2.0*(x_b[i+1] + c/b) * (std::pow(mu_1, 2) - std::pow(mu_0, 2))
                   + (c*c/(2.0*b) + b/2.0*x_b[i+1]*x_b[i+1] + c*x_b[i+1]) * (mu_1 - mu_0);
        R_top =   a * dx[i] / 2.0 * (std::pow(mu_b[m][1],2) - std::pow(mu_1,2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_b[m][1] - mu_1)
                    + c * dx[i] * (mu_b[m][1] - mu_1);
        R_bottom =   a * dx[i] / 2.0 * (std::pow(mu_0,2) -  std::pow(mu_b[m][0],2))
                    + b * dx[i] * (x_b[i] + x_b[i+1]) / 2.0 * (mu_0 - mu_b[m][0])
                    + c * dx[i] * (mu_0 - mu_b[m][0]);
        R_i[m][i] = std::fabs(R_minus) + std::fabs(R_plus) + std::fabs(R_top) + std::fabs(R_bottom);         
      }
      
//       R_i[m][i] = - (std::pow(mu_b[m][1],2) - std::pow(mu_b[m][0],2))/2.0 * (psi[m][i][1] - psi[m][i][0])
//                   - d_mu[m]*sig_t[i]*(psi[m][i][0] + psi[m][i][1])/2.0 * dx[i]
//                   + d_mu[m]*sig_s[i]/(2.0)         //weight sums to 2.0 in 1D, 4*PI in 3D
//                            *(phi[i][0] + phi[i][1])/2.0 * dx[i] 
//                   + d_mu[m]*(q[m][i][0] + q[m][i][1])/2.0 * dx[i];
    }
  /*============ END: Build cell-interior residuals ============*/
   
  /*============ Build cell-interface residuals ============*/
  for(unsigned int m=0; m<num_quads; m++)
  {
    for(unsigned int i=1; i<=num_cells-1; i++)
    {
      R_s[m][i] = - (std::pow(mu_b[m][1],2) - std::pow(mu_b[m][0],2))/2.0 * (psi[m][i][0] - psi[m][i-1][1]);
    }
    if( mu_m[m] >= 0.0 )  //if mu>0, we have left incident boundary source due to boundary condition
//      R_s[m][0] = (psi_L - psi[m][0][0])*d_mu[m];
      R_s[m][0] = - (std::pow(mu_b[m][1],2) - std::pow(mu_b[m][0],2))/2.0 * (psi[m][0][0] - psi_L);
    if( mu_m[m] <= 0.0 )  //if mu>0, we have right incident boundary source due to boundary condition
//      R_s[m][num_cells] = (psi_R - psi[m][num_cells-1][1])*d_mu[m];
      R_s[m][num_cells] = - (std::pow(mu_b[m][1],2) - std::pow(mu_b[m][0],2))/2.0 * (psi_R -  psi[m][num_cells-1][1]);
  }
  /*============ END: Build cell-interface residuals ============*/
  
  /*============ Prepare for Sampling =============*/
  // CRF = cumulative residual function, analogous to PDF
  int nRperm = 2 * num_cells + 1;  //number residuals per direction(interior or surfaces)
  std::vector<double> CRF(num_quads*nRperm, 0.0);
  double S = 0.0; //sum of all residuals for all cell interior and surfaces
  
  for(unsigned int m=0; m<num_quads; m++)
    for(unsigned int i=0; i<num_cells+1; i++)
    {
      S += std::fabs(R_s[m][i]);
      CRF[m*nRperm + i*2] = S;
      std::cout<<"CRF["<<m<<","<<i<<"]("<<m*nRperm + i*2<<") = "<<CRF[m*nRperm + i*2]<<std::endl;
      
      if(i < num_cells)
      {
        S += std::fabs(R_i[m][i]);
        CRF[m*nRperm + i*2 + 1] = S;      
        std::cout<<"CRF["<<m<<","<<i<<"]("<<m*nRperm + i*2 + 1<<") = "<<CRF[m*nRperm + i*2 + 1]<<std::endl;
      }
    }
    
    std::cout<<"S = "<<S<<std::endl;
    
  srand((unsigned int)time(NULL));    //set system time as random number seed
  double s;                           //sampled random number from uniform distribution between [0,1]
  double s_mu, s_x;                   //sampled random number for generating mu and x
  double s_rej;                       //second sampled random number in rejection method
  double x_0;                         // x_0 = arg[P(mu, x)=0] for a given mu 
  
  double a, b, c, R_minus, R_plus, R_top, R_bottom;
  
  double mu;       //sampled mu
  double x;        //sampled x
  double weight;   //sampled particle weight (+ or -)
  
  std::vector<std::vector<double> > d_phi(Nbatches, std::vector<double>(num_cells, 0.0));     //scalar flux computed by MC with residual as source
  
  //run through Nparticles histories
  for(unsigned int nb=0; nb<Nbatches; nb++)
  {
    for(unsigned int n=0; n<Nparticles; n++)
    {
      s = float(rand())/float(RAND_MAX);
      for(unsigned int j=0; j<num_quads*nRperm; j++)
      {
        if( s*S <= CRF[j] )  //found the encompassing cell or surface
        {
  //debug        std::cout<<"s= "<<s*S<<",  CRF["<<j<<"]="<<CRF[j]<<"  |  ";
          unsigned int m = int(j/nRperm);
          unsigned int i_R = j%nRperm;
          unsigned int i = i_R/2;
          unsigned int CorS = i_R%2;    //Cell or Surface
  //debug        std::cout<<"[m, i, i_surface] = "<<m<<" "<<i<<" "<<CorS<<std::endl;
        
          if( CorS == 1 ) //If the sampled particle is within a cell interior
          {
            a = - (psi[m][i][1] - psi[m][i][0]) / dx[i];
            b = (- sig_t[i] * (psi[m][i][1] - psi[m][i][0]) + sig_s[i]/2.0 * (phi[i][1] - phi[i][0])
                + (q[m][i][1] - q[m][i][0]))/dx[i];
            c = - sig_t[i] * (psi[m][i][0] - (psi[m][i][1] - psi[m][i][0])*x_b[i]/dx[i])
                + sig_s[i]/2.0 * (phi[i][0] - (phi[i][1] - phi[i][0])*x_b[i]/dx[i])
                + (q[m][i][0] - (q[m][i][1] - q[m][i][0])*x_b[i]/dx[i]);
          
            // Find the maximum value of residual within the [mu, x] domain
            // Because residual is assumed to be bilinear, extremum is obtained on corners
            double p1 = std::fabs(a*mu_b[m][0] + b*x_b[i] + c);
            double p2 = std::fabs(a*mu_b[m][1] + b*x_b[i] + c);
            double p3 = std::fabs(a*mu_b[m][1] + b*x_b[i+1] + c);
            double p4 = std::fabs(a*mu_b[m][0] + b*x_b[i+1] + c);
          
            double h = std::max({p1, p2, p3, p4});  //h is the extremum of residual
          
            while(1)  //start rejection sampling procedure
            {
              s_mu = float(rand())/float(RAND_MAX);
              mu = mu_b[m][0] + (mu_b[m][1] -  mu_b[m][0]) * s_mu;
              s_x = float(rand())/float(RAND_MAX);
              x = x_b[i] + (x_b[i+1] - x_b[i]) * s_x;
              double p = a*mu + b*x + c;
          
              s_rej = float(rand())/float(RAND_MAX);
              if( s_rej < std::fabs(p)/h)  //sampled mu and x are accepted
              {
                weight = p/std::fabs(p);
  //debug              std::cout<<"mu = "<<mu<<",  x = "<<x<<",  particle weigth = "<< weight
  //debug                       <<std::endl;
                break;
              }
              else  //sampled mu and x are rejected, repeat sampling
                continue;
            }
          
  //           x_0 = - (a*mu + c)/b;
  //           
  //           double p;  //probability (residual) for the sampled point
  //           
  //           if(x_0 <= x_b[i] || x_0 >= x_b[i])  //if p doesn't change sign in the x range for the given mu
  //             p = std::fabs(a*mu*dx[i] + b/2.0*dx[i]*(x_b[i]+x_b[i+1])/2.0 + c*dx[i]);
  //           else
  //           {
  //             R_minus = a*mu*(x_0 - x_b[i]) + b/2.0*(x_0 - x_b[i])*(x_0+x_b[i])/2.0 + c*(x_0 - x_b[i]);
  //             R_plus = a*mu*(x_b[i+1] - x_0) + b/2.0*(x_b[i+1] - x_0)*(x_0+x_b[i+1])/2.0 + c*(x_b[i+1] - x_0);
  //             p = std::fabs(R_minus) + std::fabs(R_plus);
  //           }  

            double ds;
            ds_btw_collision(sig_t[i], ds);
            RayTracing(x_b, x, ds, mu, int(i), num_cells, mu_d, wt_d, sig_t, sig_s, weight, d_phi[nb]);
  //          double x_new, mu_out;
  //           while(1)
  //           {
  //             x_new = x + ds*mu;
  //             if( i==0 && x_new<0.0 ) //leak through left boundary
  //             {
  //               d_phi[i] += std::fabs(x/mu);
  //               break;
  //             }
  //             else if( i==num_cells && x_new > x_b[num_cells] ) //leak through left boundary
  //             {
  //               d_phi[i] += std::fabs((x_b[num_cells] - x)/mu);
  //               break;
  //             }
  //             else if( i < FindCell(x_b, x_new) || x_new > x_b[num_cells] )
  //             {
  //               d_phi[i] += std::fabs((x_b[i+1] - x)/mu);
  //               x=x_b[i+1];		//bring particle to boundary
  //               i++;
  //               ds_btw_collision(sig_t[i], ds);		//sample distance between collision
  //             }
  //             else if( i > FindCell(x_b, x_new) || x_new < x_b[0] )
  //             {
  //               d_phi[i] += std::fabs((x -  x_b[i])/mu);
  //               x=x_b[i];		//bring particle to boundary
  //               i--;
  //               ds_btw_collision(sig_t[i], ds);		//sample distance between collision
  //             }
  //             else  //Collision happened in the same cell
  //             /*=============== resample post-scattering angle ========================*/
  //             {
  //               d_phi[i] += ds;
  //               s = float(rand())/float(RAND_MAX);
  //               if(s <= (sig_t[i] - sig_s[i])/sig_t[i])		//if absorbed, terminate current history
  //                 break;
  //               else
  //               {
  //                 x=x_new;				//confirm x postion after collision
  //                 scat_angle_lab(mu, mu_d, wt_d, mu_out);  //sample post-scatter mu in lab frame
  //                 mu = mu_out;
  //                 ds_btw_collision(sig_t[i], ds);		//sample distance between collision
  //               }
  //             }
  //           }

          }
          else if( CorS == 0 ) //If the sampled particle is at a cell surface
          {          
//debug            std::cout<<"[m, i, CellorSurface] = "<<m<<" "<<i<<" "<<CorS<<std::endl;
            double mu_sign = 1.0;
            // 1). determine the sign of mu
            if( (mu_b[m][0] + mu_b[m][1])/2.0 < 0 )
              mu_sign = -1.0;
            // 2). determine the absolute value of mu
            s_mu = float(rand())/float(RAND_MAX);
            mu = sqrt(s_mu*(std::pow(mu_b[m][1],2) - std::pow(mu_b[m][0],2)) + std::pow(mu_b[m][0],2)); 
            mu = mu_sign*mu;
            x = x_b[i];
            // 3). determine the sign of the weight of the particle
            weight = R_s[m][i]/std::fabs(R_s[m][i]);
//debug          std::cout<<"mu = "<<mu<<",  x = "<<x<<",  particle weigth = "<< weight
//debug                   << " | Surface" <<std::endl;
                   
            double ds;
            //determine which cell the particle is born in
            if( mu_sign > 0.0 && i!=num_cells) //if mu>0, then particle traces out in cell on the right
              i = i;
            if( mu_sign > 0.0 && i==num_cells)
              i = i-1;
            if( mu_sign < 0.0 && i!=0) //if mu<0, then particle traces out in cell on the left
              i = i-1;
            ds_btw_collision(sig_t[i], ds);
            RayTracing(x_b, x, ds, mu, int(i), num_cells, mu_d, wt_d, sig_t, sig_s, weight, d_phi[nb]);
          }
        
          break;
        }
        else  //keep looking for encompassing cell or surface
          continue;
      }
    }
    //output d_phi computed by MC
    std::cout<<"Batch #"<<nb<<": "<<std::endl;
    for(unsigned int i=0; i<num_cells; i++)
      std::cout<<"d_Phi["<<i<<"] = "<<d_phi[nb][i]/Nparticles*S<<std::endl;
  }
  
  std::vector<double> d_phi_mean(num_cells, 0.0);
  std::vector<double> d_phi_sq_mean(num_cells, 0.0);
  std::vector<double> sd_d_phi(num_cells, 0.0);
  
  //statistics
  std::cout<<"========= Statistics ==========="<<std::endl;
  for(unsigned int nb=0; nb<Nbatches; nb++)
  {
    for(unsigned int i=0; i<num_cells; i++)
    {
      d_phi_mean[i] += d_phi[nb][i]/Nparticles*S/Nbatches;
      d_phi_sq_mean[i] += pow(d_phi[nb][i]/Nparticles*S, 2)/Nbatches;
    }
  }
  
  for(unsigned int i=0; i<num_cells; i++)
  {
    std::cout<<"d_Phi_mean["<<i<<"] = "<<d_phi_mean[i]<<std::endl;
  }
  for(unsigned int i=0; i<num_cells; i++)
  {
    sd_d_phi[i] = sqrt( Nbatches/(Nbatches - 1) * (d_phi_sq_mean[i] - pow(d_phi_mean[i], 2)) );
    std::cout<<"SD["<<i<<"] = "<< sd_d_phi[i]<<std::endl;
  }
  
  
  
  return 0;
}