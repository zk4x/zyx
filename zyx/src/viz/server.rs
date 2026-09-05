// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! HTTP server for the graph visualizer (`tiny_http`, background thread).
//!
//! Routes:
//! - `/`                    — the web page
//! - `/vis-network.min.js`  — vendored graph drawing library
//! - `/api/graphs`          — JSON list of realized graphs (tabs)
//! - `/api/graph/<id>`      — JSON plan structure of one graph
//! - `/api/kernel/<g>/<k>/<stage>` — sched | ir | asm?target=<target>
use super::{Target, VizData};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Response, Server};

/// Serve the visualizer until process exit.
pub(super) fn spawn(data: Arc<Mutex<VizData>>) {
    std::thread::Builder::new()
        .name("zyx-viz".to_string())
        .spawn(move || run(data))
        .expect("failed to spawn zyx-viz server thread");
}

fn run(data: Arc<Mutex<VizData>>) {
    let server = match Server::http("0.0.0.0:4242") {
        Ok(server) => server,
        Err(e) => {
            eprintln!("[viz] failed to bind 0.0.0.0:4242: {e}");
            return;
        }
    };
    println!("[viz] graph visualizer serving at http://0.0.0.0:4242");

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let (path, query) = match url.split_once('?') {
            Some((path, query)) => (path.to_string(), Some(query.to_string())),
            None => (url, None),
        };
        let response = route(&data, &path, query.as_deref());
        let _ = request.respond(response);
    }
}

fn html(content_type: &'static str, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(body).with_header(Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes()).unwrap())
}

fn route(data: &Arc<Mutex<VizData>>, path: &str, query: Option<&str>) -> Response<std::io::Cursor<Vec<u8>>> {
    match path {
        "/" | "/index.html" => return html("text/html; charset=utf-8", super::page::INDEX_HTML.to_string()),
        "/vis-network.min.js" => {
            return Response::from_data(super::page::VIS_NETWORK_JS)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/javascript"[..]).unwrap());
        }
        "/api/graphs" => {
            let d = data.lock().unwrap();
            let graphs: Vec<String> = d
                .graphs
                .iter()
                .enumerate()
                .map(|(i, g)| format!("{{\"id\":{i},\"name\":{},\"kernels\":{}}}", escape(&g.name), g.kernels.len()))
                .collect();
            return html("application/json", format!("[{}]", graphs.join(",")));
        }
        _ => {}
    }

    if let Some(rest) = path.strip_prefix("/api/graph/") {
        let Ok(id) = rest.parse::<usize>() else {
            return html("text/plain; charset=utf-8", "bad graph id".to_string());
        };
        let d = data.lock().unwrap();
        let Some(g) = d.graphs.get(id) else {
            return html("text/plain; charset=utf-8", "no such graph".to_string());
        };
        return html("application/json", graph_json(g));
    }

    if let Some(rest) = path.strip_prefix("/api/kernel/") {
        let parts: Vec<&str> = rest.split('/').collect();
        let [g, k, stage] = parts[..] else {
            return html("text/plain; charset=utf-8", "expected /api/kernel/<g>/<k>/<stage>".to_string());
        };
        let (Ok(g), Ok(k)) = (g.parse::<usize>(), k.parse::<usize>()) else {
            return html("text/plain; charset=utf-8", "bad indices".to_string());
        };

        // Copy the capture out, then derive without holding the lock.
        let cap = match (|| {
            let d = data.lock().unwrap();
            let Some(graph) = d.graphs.get(g) else {
                return Err("no such graph");
            };
            match graph.kernels.get(k) {
                Some(Some(cap)) => Ok(cap.clone()),
                Some(None) => Err("AOT kernel has no captured IR"),
                None => Err("no such kernel"),
            }
        })() {
            Ok(cap) => cap,
            Err(msg) => return html("text/plain; charset=utf-8", msg.to_string()),
        };

        let body = match stage {
            "sched" => cap.sched_kernel.render(false),
            "ir" => super::derive_optimized(&cap).render(false),
            "asm" => {
                let target =
                    query.and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("target="))).and_then(Target::from_str);
                match target {
                    Some(target) => super::generate_source(&cap, target),
                    None => format!(
                        "unknown target; available: {}",
                        Target::ALL.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(", ")
                    ),
                }
            }
            _ => "unknown stage; expected sched, ir or asm".to_string(),
        };
        return html("text/plain; charset=utf-8", body);
    }

    html("text/plain; charset=utf-8", "not found".to_string())
}

fn graph_json(g: &super::GraphViz) -> String {
    let nodes: Vec<String> =
        g.nodes.iter().map(|n| format!("{{\"id\":{},\"label\":{},\"kernel\":{}}}", n.id, escape(&n.label), n.kernel)).collect();
    let edges: Vec<String> = g.edges.iter().map(|(f, t, l)| format!("[{f},{t},{}]", escape(l))).collect();
    let devices: Vec<String> = g
        .kernels
        .iter()
        .map(|k| match k {
            Some(cap) => escape(cap.device_label),
            None => "null".to_string(),
        })
        .collect();
    format!("{{\"nodes\":[{}],\"edges\":[{}],\"devices\":[{}]}}", nodes.join(","), edges.join(","), devices.join(","))
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
