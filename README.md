# SandBox
This is my SandBox for computer graphics

## Architecture
```
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
---

## Work In Progress

### ✅ Completed
* **초기 initialization 설정 (완료)**
  * Window 인터페이스: GLFW 및 GLAD를 활용한 윈도우 컨텍스트 생성 및 Window 클래스 래핑 완료.
  * 애플리케이션 프레임워크: 메인 루프(Update/Render)를 담당하는 App 클래스 및 전체 프로젝트 폴더 구조(src/core, render, scene 등) 확립.

* **GPU 연동 및 디스플레이 (완료)**
  * CUDA + OpenGL Interop: PBO(Pixel Buffer Object)를 활용하여 CUDA 커널의 계산 결과를 OpenGL 텍스처로 실시간 전송 및 출력하는 파이프라인 구축.
  * 검증 커널: TestKernel.cu를 통해 화면에 그라데이션을 띄워 데이터 전송 무결성 확인.

* **OptiX 환경 설정 및 커널 로드 (완료)**
  * 빌드 시스템: CMake를 통해 OptiX 8.0 SDK 및 관련 종속성(CUDA Toolkit 등) 링크 완료.
  * 컨텍스트 초기화: RenderContext 클래스에서 optixDeviceContextCreate를 포함한 기본 디바이스 초기화 로직 구현.
  * Module 생성: PathTracingKernel.ptx 파일을 로드하여 OptiX Module 생성 완료.
  * Program Group 생성: Raygen, Miss, HitGroup(ClosestHit) 커널을 각각 연결하여 OptixProgramGroup 구성 완료.

### 🚀 To-Do (Next Steps)
* **Pipeline 및 SBT 구축 (진행 중)**
  * Pipeline 생성: 생성된 Module과 Program Group들을 연결하여 OptiX Pipeline (optixPipelineCreate) 구축.
  * SBT(Shader Binding Table) 구현: Raygen, Miss, HitGroup 레코드를 호스트 메모리에서 구성하고, CUDA 디바이스 메모리에 할당하여 매핑.

* **Basic Launch**
  * 기존 CUDA 그라데이션 커널을 optixLaunch로 대체하여 화면 출력 확인.

* **Scene & Asset Loading**
  * Asset Loader: tinyobjloader를 프로젝트(extern/)에 클론 및 포함시키고 메시 로드 로직 구현.
  * AS(Acceleration Structure) 빌드: 로드된 메시 데이터(Vertex/Index)를 GPU 버퍼로 전송하고 GAS/IAS 구축.

* **Camera System**
  * Camera.h를 완성하여 마우스/키보드 입력에 따른 실시간 뷰 매트릭스 변환 구현.

* **Integrator 모듈화**
  * PathTracer, BDPT 등 다양한 알고리즘을 인터페이스화하여 메인 루프에서 교체 가능한 구조 완성.

* **GUI 통합**
  * Dear ImGui를 연동하여 렌더링 파라미터(SPP, 루프 횟수 등) 실시간 제어.