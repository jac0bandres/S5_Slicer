"""Compare the Python C++-convention path with the headless native oracle."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from s3 import read_tet, write_tet
from s3.deform import support_free


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--fixture',type=Path,default=Path(__file__).parents[1]/'tests/fixtures/cantilever.tet')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    mesh=read_tet(args.fixture,cpp_float32=True)
    points=mesh.points[:,[0,2,1]].copy();points[:,2]*=-1
    with tempfile.TemporaryDirectory(prefix='s3-native-') as directory:
        directory=Path(directory)
        source=directory/'fixture.tet';prefix=directory/'native'
        write_tet(source,points,mesh.cells)
        completed=subprocess.run([str(args.executable.resolve()),str(source),str(prefix)],
                                  capture_output=True,text=True,check=True)
        result=support_free(read_tet(source,cpp_float32=True))
        errors={}
        for key,suffix in [('heat','heat'),('growing','growing'),('scales','scales'),('deformed','points')]:
            actual=result[key];expected=np.loadtxt(str(prefix)+'-'+suffix+'.txt')
            if key=='deformed':
                actual=actual-actual.mean(axis=0)
                expected=expected-expected.mean(axis=0)
            np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7,err_msg=key)
            errors[key]=float(np.max(np.abs(actual-expected)))
        report=dict(fixture=str(args.fixture),max_absolute_errors=errors,
                    comparison='active C++ support-free functions; sparse-backend and translation-gauge adaptations',
                    native_returncode=completed.returncode)
        if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
