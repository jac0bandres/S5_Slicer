// Harness only; numerical implementations are extracted from upstream at build.
#include <fstream>
#include <iostream>
#include <iomanip>
#include "DeformTet.h"

int main(int argc, char** argv) {
 if(argc!=3) { std::cerr << "usage: reference input.tet output-prefix\n"; return 2; }
 QMeshPatch mesh;
 if(!mesh.inputTETFile(argv[1],false)) return 3;
 PolygenMesh poly; poly.meshes.AddTail(&mesh);
 DeformTet solver;
 solver.initial(&poly,2,7,15.,0.,0.,0.,5.,1.,40.,1);
 solver.m_supportFreeAngle=30.;
 solver._cal_heatMethod();
 std::ofstream heat(std::string(argv[2])+"-heat.txt"); heat << std::setprecision(17);
 for(auto pos=mesh.GetNodeList().GetHeadPosition();pos;) {
  auto node=(QMeshNode*)mesh.GetNodeList().GetNext(pos);
  heat << node->HeatFieldValue << "\n";
 }
 std::ofstream grow(std::string(argv[2])+"-growing.txt"); grow << std::setprecision(17);
 for(auto pos=mesh.GetTetraList().GetHeadPosition();pos;) {
  auto tet=(QMeshTetra*)mesh.GetTetraList().GetNext(pos);
  grow << tet->vectorField_4voxelOrder.transpose() << "\n";
 }
 solver.runASAP_SupportLess_test3();
 std::ofstream out(std::string(argv[2])+"-points.txt"); out << std::setprecision(17);
 for(auto pos=mesh.GetNodeList().GetHeadPosition();pos;) {
  auto node=(QMeshNode*)mesh.GetNodeList().GetNext(pos);
  double x,y,z; node->GetCoord3D(x,y,z); out << x << " " << y << " " << z << "\n";
 }
 std::ofstream scales(std::string(argv[2])+"-scales.txt"); scales << std::setprecision(17);
 for(auto pos=mesh.GetTetraList().GetHeadPosition();pos;) {
  auto tet=(QMeshTetra*)mesh.GetTetraList().GetNext(pos);
  scales << tet->scaleValue_vector.transpose() << "\n";
 }
 poly.meshes.RemoveAll();
 return 0;
}
