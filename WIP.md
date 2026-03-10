### Work In Progress

* 초기 initalization 설정 (완료)
  * Window 인터페이스: GLFW 및 GLAD를 활용한 윈도우 컨텍스트 생성 및 Window 클래스 래핑 완료.
  * 애플리케이션 프레임워크: 메인 루프(Update/Render)를 담당하는 App 클래스 및 전체 프로젝트 폴더 구조(src/core, render, scene 등) 확립.

* GPU 연동 및 디스플레이 (완료)
  * CUDA + OpenGL Interop: PBO(Pixel Buffer Object)를 활용하여 CUDA 커널의 계산 결과를 OpenGL 텍스처로 실시간 전송 및 출력하는 파이프라인 구축.
  * 검증 커널: TestKernel.cu를 통해 화면에 그라데이션을 띄워 데이터 전송 무결성 확인.

* OptiX 환경 설정 (완료)
  * 빌드 시스템: CMake를 통해 OptiX 8.0 SDK 및 관련 종속성(CUDA Toolkit 등) 링크 완료.
  * 컨텍스트 초기화: RenderContext 클래스에서 optixDeviceContextCreate를 포함한 기본 디바이스 초기화 로직 구현.

* SBT(Shader Binding Table) 구현: Raygen, Miss, HitGroup 레코드 구성.
 
* Module & Pipeline 생성: PathTracingKernel.cu를 컴파일하여 OptiX 파이프라인 구축.

* Basic Launch: 기존 CUDA 그라데이션 커널을 optixLaunch로 대체하여 화면 출력.

* Asset Loader: tinyobjloader를 활용한 메시 로드 로직 구현.

* AS(Acceleration Structure) 빌드: 로드된 메시 데이터를 GPU로 전송하고 GAS/IAS 구축.

* Camera System: Camera.h를 완성하여 마우스 입력에 따른 실시간 뷰 변환 구현.

* Integrator 모듈화: PathTracer, BDPT 등 다양한 알고리즘을 인터페이스화하여 교체 가능한 구조 완성.

* GUI 통합: Dear ImGui를 연동하여 렌더링 파라미터(SPP, 루프 횟수 등) 실시간 제어.