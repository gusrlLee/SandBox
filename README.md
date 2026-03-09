# SandBox
This is my SandBox for computer graphics

# architecture 
Zero/
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