# SandBox
This is my SandBox for computer graphics

## Architecture
```text
SandBox/
├── CMakeLists.txt           # 전체 빌드 설정 (OptiX, CUDA, C++ 설정)
├── extern/                  # 외부 라이브러리 (수정하지 않는 코드)
│   ├── imgui/
│   ├── tinyobjloader/
│   ├── tinygltf/
│   └── glfw/
├── assets/                  # 렌더링할 모델(.obj, .gltf) 및 텍스처
├── src/
│   ├── main.cpp             # 진입점, Window 및 App 초기화
│   ├── core/                # 어플리케이션 뼈대
│   │   ├── Application.h/cpp # 메인 루프 (Update, Render, GUI 처리)
│   │   ├── Window.h/cpp      # GLFW 윈도우 관리 및 입력(키보드/마우스)
│   │   └── Camera.h/cpp      # Interactive User Camera (마우스로 움직이는 로직)
│   ├── scene/               # 씬 데이터 (OptiX와 독립적인 순수 데이터)
│   │   ├── Scene.h/cpp       # Mesh, Light, Material 등을 담는 컨테이너
│   │   ├── Mesh.h/cpp        # tinyobj/tinygltf 로더 래핑
│   │   └── Material.h        # BSDF 파라미터 구조체
│   ├── render/              # 렌더링 백엔드 (OptiX, CUDA, OpenGL Interop)
│   │   ├── RenderContext.h/cpp   # OptiX Device, Pipeline, SBT 초기화 및 관리
│   │   ├── GLInterop.h/cpp       # CUDA 렌더링 결과를 OpenGL 텍스처로 띄우는 역할 (PBO)
│   │   └── buffer.h              # CUDA 디바이스 메모리 래퍼 (할당/해제 관리)
│   ├── integrators/         # ★ 논문 구현 및 연구의 핵심 폴더 (모듈화)
│   │   ├── Integrator.h      # 모든 렌더링 기법의 부모 인터페이스 (virtual render())
│   │   ├── PathTracer.h/cpp
│   │   ├── PhotonMapper.h/cpp
│   │   └── BDPT.h/cpp
│   └── shaders/             # OptiX 커널 및 CUDA 디바이스 코드 (.cu, .h)
│       ├── common.h          # 호스트(C++)와 디바이스(CUDA)가 공유하는 구조체 (Ray Payload 등)
│       ├── pt_kernels.cu     # Path Tracing용 Raygen, Miss, ClosestHit
│       └── bdpt_kernels.cu   # BDPT용 커널
```

📊 Current Progress: ~50%
가장 난이도가 높은 GPU Interop 및 OptiX 렌더링 파이프라인 기반 구축 완료. 이제 씬 데이터 로드 및 실제 렌더링 알고리즘 구현 단계로 진입.

Work In Progress
✅ Completed
초기 initialization 설정

Window 인터페이스: GLFW 및 GLAD를 활용한 윈도우 컨텍스트 생성 및 Window 클래스 래핑 완료.

애플리케이션 프레임워크: 메인 루프(Update/Render)를 담당하는 App 클래스 및 전체 프로젝트 폴더 구조 확립.

GPU 연동 및 디스플레이

CUDA + OpenGL Interop: PBO를 활용하여 CUDA 커널의 계산 결과를 OpenGL 텍스처로 실시간 전송하는 파이프라인 구축.

OptiX 환경 설정 및 파이프라인 구축

빌드 시스템: CMake를 통해 OptiX 8.0 SDK 및 CUDA Toolkit 링크.

컨텍스트 초기화 및 커널 로드: Context, Module(PTX 로드), Program Group 구성 완료.

Pipeline 및 SBT 구축: Pipeline 생성 및 Shader Binding Table 디바이스 메모리 할당 및 매핑 성공.

Basic Launch (First Pixel)

optixLaunch 연동: 디바이스 파라미터 버퍼 업데이트 및 Raygen 커널을 통한 PBO 그라데이션 화면 출력 성공.

🚀 To-Do (Next Steps)
Scene & Asset Loading (진행 중)

Asset Loader: tinyobjloader를 프로젝트에 포함시키고, .obj 파일을 읽어 정점(Vertex)과 인덱스(Index) 데이터를 파싱하는 Mesh 클래스 로직 구현.

Acceleration Structure (AS) 빌드

로드된 Mesh 데이터를 GPU 버퍼로 전송.

OptiX 하드웨어 가속을 위한 GAS(Geometry Acceleration Structure) 및 IAS(Instance Acceleration Structure) 구축.

Camera System

Camera.h를 완성하여 마우스/키보드 입력에 따른 실시간 뷰 매트릭스 변환 로직 구현.

Integrator & Ray Tracing 로직

ClosestHit 커널(pt_kernels.cu) 업데이트: 광선과 Mesh가 교차했을 때 법선(Normal)이나 색상을 출력하도록 수정.

Integrator 모듈화: PathTracer 인터페이스 설계 및 적용.

GUI 통합

Dear ImGui를 연동하여 샘플링 수(SPP), 재질 파라미터 등을 런타임에 실시간으로 제어.

🔮 Future Roadmap (Advanced)
Materials & Textures: BSDF 파라미터 구조체를 커널에 전달하여 다양한 재질(Diffuse, Specular, Glass 등) 표현 및 텍스처 매핑 지원.

Lighting: Area Light 샘플링 및 HDRI 환경 맵(Environment Mapping) 조명 구현.

Advanced Integrators: 논문 연구를 위한 BDPT(Bidirectional Path Tracing) 및 Photon Mapping 구현.

