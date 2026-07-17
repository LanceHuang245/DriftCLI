use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use drift_config::{McpConfig, McpServerConfig, McpTransport};
use drift_mcp::McpManager;
use drift_security::{PermissionEngine, ProcessSandbox, SandboxMode, SecurityConfig};
use drift_tools::tools::read::ReadTool;
use drift_tools::{ToolContext, ToolRegistry};
use tokio::sync::mpsc;

fn marker_path() -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock must be valid")
        .as_nanos();
    std::env::temp_dir().join(format!("drift-mcp-fixture-{suffix}.marker"))
}

fn unrestricted_sandbox() -> Arc<ProcessSandbox> {
    let cwd = std::env::current_dir().expect("cwd must exist");
    Arc::new(ProcessSandbox::new(SandboxMode::DangerFullAccess, &cwd).unwrap())
}

#[tokio::test]
async fn stdio_servers_register_call_and_shutdown() {
    let marker = marker_path();
    let _ = std::fs::remove_file(&marker);
    let fixture = McpServerConfig {
        id: "fixture".into(),
        command: env!("CARGO_BIN_EXE_drift-mcp-test-server").into(),
        args: Vec::new(),
        env: [(
            "DRIFT_MCP_FIXTURE_EXIT_FILE".into(),
            marker.to_string_lossy().into(),
        )]
        .into_iter()
        .collect(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let invalid = McpServerConfig {
        id: "broken".into(),
        command: "drift-command-that-does-not-exist".into(),
        args: Vec::new(),
        env: Default::default(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let duplicate = McpServerConfig {
        id: "fixture".into(),
        command: "duplicate-must-not-start".into(),
        args: Vec::new(),
        env: Default::default(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let mut registry = ToolRegistry::new();
    registry.register_builtin(Arc::new(ReadTool));
    let registry = Arc::new(registry);
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let manager = Arc::new(McpManager::with_status_sender(
        McpConfig {
            enabled: true,
            servers: vec![fixture, duplicate, invalid],
        },
        registry.clone(),
        status_tx,
        unrestricted_sandbox(),
    ));

    manager.clone().start_auto_servers().await;

    let mut statuses = Vec::new();
    while let Ok(status) = status_rx.try_recv() {
        statuses.push(status);
    }
    assert!(
        statuses
            .iter()
            .any(|(id, status)| id == "fixture" && status.starts_with("connected"))
    );
    assert!(
        statuses
            .iter()
            .any(|(id, status)| id == "broken" && status.starts_with("failed:"))
    );
    assert!(
        statuses
            .iter()
            .any(|(id, status)| { id == "fixture" && status.contains("duplicate server id") })
    );

    let tool = registry
        .get_async("mcp__fixture__echo")
        .await
        .expect("fixture tool must be registered");
    let security = SecurityConfig::default();
    let engine = PermissionEngine::new(&security, "default");
    let cwd = std::env::current_dir().expect("cwd must exist");
    let ctx = ToolContext {
        session_id: uuid::Uuid::nil(),
        working_dir: cwd.clone(),
        tool_call_id: "test-call".into(),
        file_access: Arc::new(engine.file_access_guard(&cwd).expect("guard must build")),
        network: Arc::new(engine.network_guard()),
        process_sandbox: unrestricted_sandbox(),
    };
    let result = tool
        .execute(serde_json::json!({"text": "hello"}), &ctx)
        .await
        .expect("fixture tool call must succeed");
    assert!(result.success);
    assert_eq!(result.content, "hello");

    manager.shutdown().await;
    assert!(registry.get_async("mcp__fixture__echo").await.is_none());
    assert!(registry.get_async("read").await.is_some());
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if marker.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("fixture child must exit after shutdown");
    assert!(marker.exists());
    let _ = std::fs::remove_file(marker);
}

#[tokio::test]
async fn shutdown_cancels_in_progress_startup() {
    let fixture = McpServerConfig {
        id: "slow-fixture".into(),
        command: env!("CARGO_BIN_EXE_drift-mcp-test-server").into(),
        args: Vec::new(),
        env: [("DRIFT_MCP_FIXTURE_HANG_INITIALIZE".into(), "1".into())]
            .into_iter()
            .collect(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let registry = Arc::new(ToolRegistry::new());
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let manager = Arc::new(McpManager::with_status_sender(
        McpConfig {
            enabled: true,
            servers: vec![fixture],
        },
        registry.clone(),
        status_tx,
        unrestricted_sandbox(),
    ));
    let start_task = tokio::spawn(manager.clone().start_auto_servers());

    let status = tokio::time::timeout(Duration::from_secs(1), status_rx.recv())
        .await
        .expect("startup status must arrive")
        .expect("status channel must remain open");
    assert_eq!(status, ("slow-fixture".into(), "connecting".into()));

    manager.shutdown().await;
    tokio::time::timeout(Duration::from_secs(1), start_task)
        .await
        .expect("startup task must stop promptly")
        .expect("startup task must not panic");
    assert!(
        registry
            .get_async("mcp__slow-fixture__echo")
            .await
            .is_none()
    );
}

#[tokio::test]
async fn provider_invalid_tool_name_fails_without_registration() {
    let fixture = McpServerConfig {
        id: "invalid-tool".into(),
        command: env!("CARGO_BIN_EXE_drift-mcp-test-server").into(),
        args: Vec::new(),
        env: [("DRIFT_MCP_FIXTURE_TOOL_NAME".into(), "bad.name".into())]
            .into_iter()
            .collect(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let registry = Arc::new(ToolRegistry::new());
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let manager = Arc::new(McpManager::with_status_sender(
        McpConfig {
            enabled: true,
            servers: vec![fixture],
        },
        registry.clone(),
        status_tx,
        unrestricted_sandbox(),
    ));

    manager.clone().start_auto_servers().await;
    let statuses = std::iter::from_fn(|| status_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(
        statuses
            .iter()
            .any(|(_, status)| status.contains("tool name is not provider-safe"))
    );
    assert!(
        registry
            .get_async("mcp__invalid-tool__bad.name")
            .await
            .is_none()
    );
    manager.shutdown().await;
}

#[tokio::test]
async fn read_only_sandbox_does_not_start_mcp_processes() {
    let server = McpServerConfig {
        id: "read-only".into(),
        command: "this-command-must-not-be-resolved".into(),
        args: Vec::new(),
        env: Default::default(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let registry = Arc::new(ToolRegistry::new());
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let cwd = std::env::current_dir().expect("cwd must exist");
    let sandbox = Arc::new(ProcessSandbox::new(SandboxMode::ReadOnly, &cwd).unwrap());
    let manager = Arc::new(McpManager::with_status_sender(
        McpConfig {
            enabled: true,
            servers: vec![server],
        },
        registry,
        status_tx,
        sandbox,
    ));

    // Read-only mode reports the disabled server without resolving or spawning its command.
    manager.start_auto_servers().await;

    assert_eq!(
        status_rx.try_recv().unwrap(),
        ("read-only".into(), "disabled: read-only sandbox".into())
    );
}
