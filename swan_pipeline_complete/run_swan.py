import subprocess
import sys
from config import DOCKER_PLATFORM, PROCESSED_DIR, SWAN_DOCKER_IMAGE, SWAN_EXECUTABLE, ensure_directories
from generate_boundary import generate_boundary
from generate_depth import generate_depth
from generate_input import generate_input
from generate_wind import generate_wind

def build_pipeline():
    ensure_directories(); generate_depth(); generate_wind(); generate_boundary(); generate_input()

def run_swan():
    workdir=PROCESSED_DIR.resolve(); cmd=['docker','run','--rm']
    if DOCKER_PLATFORM: cmd += ['--platform',DOCKER_PLATFORM]
    cmd += ['-v',f'{workdir}:/work','-w','/work',SWAN_DOCKER_IMAGE]
    if SWAN_EXECUTABLE: cmd.append(SWAN_EXECUTABLE)
    print('Executando:'); print(' '.join(cmd))
    try:
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=1800,check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("O comando 'docker' não foi encontrado.") from exc
    if result.stdout: print('--- STDOUT ---'+result.stdout)
    if result.stderr: print('--- STDERR ---'+result.stderr)
    if result.returncode!=0: raise RuntimeError(f'Container/SWAN terminou com código {result.returncode}.')
    out=workdir/'output.mat'; print(f'Saída: {out}' if out.exists() else 'Aviso: output.mat não encontrado; confira o arquivo PRINT.')

def main():
    try: build_pipeline(); run_swan(); return 0
    except Exception as exc: print(f'ERRO: {exc}',file=sys.stderr); return 1

if __name__=='__main__': raise SystemExit(main())
