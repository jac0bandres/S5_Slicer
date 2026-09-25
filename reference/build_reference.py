"""Build a headless oracle from selected *actual* upstream C++ functions.

Copies go to the requested build directory; the upstream checkout is untouched.
Only GUI dependencies, sparse backend, and translation gauge handling change.
The original source license is copied with the extracted implementation.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess


def extract_function(source, name):
    start = source.index(name+'(')
    start = source.rfind('\n',0,start)+1
    # Locate the next top-level method declaration; method bodies are unchanged.
    rest = source[start:]
    matches = list(re.finditer(r'\n(?:void|bool|int|double|Eigen::\w+|QMeshFace\*) DeformTet::',rest))
    return rest[:matches[0].start()] if matches else rest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=Path)
    parser.add_argument('build',type=Path)
    args=parser.parse_args()
    root=args.upstream.resolve(); build=args.build.resolve()
    build.mkdir(parents=True,exist_ok=True)
    for name in ['QMeshLib','GLKLib','ShapeLab']:
        (build/name).mkdir(exist_ok=True)
    for name in ['QMeshNode','QMeshEdge','QMeshFace','QMeshTetra','QMeshPatch']:
        for ext in ['h','cpp']:
            text=(root/f'QMeshLib/{name}.{ext}').read_text(encoding='utf-8-sig')
            text=text.replace('#include "StdAfx.h"','#include "port_macros.h"')
            (build/f'QMeshLib/{name}.{ext}').write_text(text)
    for name in ['GLKObList','GLKGeometry','GLKMatrixLib']:
        for ext in ['h','cpp']:
            text=(root/f'GLKLib/{name}.{ext}').read_text(encoding='utf-8-sig')
            text=text.replace('#include "stdafx.h"','')
            text=text.replace('#include <../ThirdPartyDependence/eigen3/Eigen/Dense>', '#include <Eigen/Dense>')
            (build/f'GLKLib/{name}.{ext}').write_text(text)
    (build/'GLKLib/GLKLib.h').write_text('#pragma once\n#include <iostream>\n#include "GLKObList.h"\nclass QMeshNode;\n')
    macros=(root/'QMeshLib/stdafx.h').read_text()
    start=macros.index('#define CROSS')
    end=macros.index('\n',macros.index('#define DOT'))
    (build/'QMeshLib/port_macros.h').write_text(macros[start:end]+'\n')
    (build/'QMeshLib/PolygenMesh.h').write_text('''#pragma once
#include <string>
#include "QMeshPatch.h"
#include "QMeshNode.h"
#include "QMeshEdge.h"
#include "QMeshFace.h"
#include "QMeshTetra.h"
struct PolygenMesh {
 GLKObList meshes;
 GLKObList& GetMeshList() { return meshes; }
 std::string getModelName() { return "reference"; }
};
''')
    header=(root/'ShapeLab/DeformTet.h').read_text()
    header=header.replace('private:','public:').replace('void DeformTet::update_', 'void update_')
    (build/'ShapeLab/DeformTet.h').write_text(header)
    for name in ['heatMethod.h','heatMethod.cpp']:
        text=(root/'ShapeLab'/name).read_text()
        text=text.replace('#include<Eigen/PardisoSupport>','#include<Eigen/SparseLU>')
        text=text.replace('Eigen::PardisoLU','Eigen::SparseLU')
        (build/'ShapeLab'/name).write_text(text)
    source=(root/'ShapeLab/DeformTet.cpp').read_text()
    names=['initial','_index_initial','_build_tetraSet_4SpeedUp','_moveModelup2ZeroHeight',
           '_record_initial_coord3D','_record_neighbor_Tet','_record_initial_normal3D',
           '_compTetMeshVolumeMatrix','_calculate_edgeLength','_detectBottomTet',
           '_detect_each_neighbor_Tet','_cal_heatMethod','runASAP_SupportLess_test3',
           '_detectOverhangFace','_calFabricationEnergy_SupportLess',
           '_get_energy_innerLoop_supportLess','_globalQuaternionSmooth1_supportLess',
           '_get_initial_overhang_faceNormal','_cal_rotationMatrix_supportLess','_cal_virtual_Vfield']
    parts=['#include <Eigen/Eigen>\n#include <Eigen/SparseCholesky>\n#include "DeformTet.h"\n'
           '#include "heatMethod.h"\n#include "../GLKLib/GLKGeometry.h"\n'
           'DeformTet::DeformTet() {}\nDeformTet::~DeformTet() {}\n']
    for name in names:
        function=extract_function(source,'DeformTet::'+name)
        function=function.replace('Eigen::PardisoLDLT','Eigen::SimplicialLDLT')
        # The unanchored upstream normal equations have a translation nullspace.
        # One scalar constraint fixes its arbitrary representative, not its shape.
        for axis in 'xyz':
            needle=f'Solver_ASAP_4{axis}.compute(matATA_4{axis});'
            function=function.replace(needle,f'matATA_4{axis}.coeffRef(0,0) += 1.0;\n\t\t'+needle)
        parts.append(function)
    (build/'ShapeLab/DeformTet.cpp').write_text('\n'.join(parts))
    shutil.copyfile(root/'LICENSE',build/'LICENSE.upstream')
    shutil.copyfile(Path(__file__).with_name('main.cpp'),build/'main.cpp')
    sources=[build/'main.cpp',build/'ShapeLab/DeformTet.cpp',build/'ShapeLab/heatMethod.cpp']
    sources += list((build/'QMeshLib').glob('*.cpp'))+list((build/'GLKLib').glob('*.cpp'))
    command=['g++','-std=c++17','-O1','-w','-fpermissive','-ffunction-sections','-fdata-sections',
             '-I'+str(root/'ThirdPartyDependence/eigen3'),'-I'+str(build/'QMeshLib'),
             '-I'+str(build/'GLKLib'),'-I'+str(build/'ShapeLab'),
             *map(str,sources),'-Wl,--gc-sections','-o',str(build/'s3-reference')]
    subprocess.run(command,check=True)
    print(build/'s3-reference')


if __name__=='__main__': main()
