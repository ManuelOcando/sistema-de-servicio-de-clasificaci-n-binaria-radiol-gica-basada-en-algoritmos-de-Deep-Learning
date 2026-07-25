"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import MedidorConfianza from "@/components/MedidorConfianza";
import PanelModelo from "@/components/PanelModelo";
import VisorRadiografia from "@/components/VisorRadiografia";

/**
 * Umbral de derivacion a revision humana.
 *
 * Procede del analisis de calibracion sobre las 900 imagenes de prueba
 * reservadas: ninguno de los 10 errores del modelo supero una confianza de
 * 0,99, de modo que este umbral habria capturado la totalidad de los fallos
 * revisando solo el 5,8% de los casos.
 */
const UMBRAL_REVISION = 0.99;

type Respuesta = {
  prediction: { label: string; confidence: number; class_id: number };
  explainability: { heatmap_base64: string; overlay_base64: string; description: string };
  performance: {
    preprocess_time_ms: number;
    inference_time_ms: number;
    explainability_time_ms: number;
    total_latency_ms: number;
    model_used: string;
  };
};

const FORMATOS = ["image/jpeg", "image/png", "image/webp"];
const TAMANO_MAXIMO = 8 * 1024 * 1024;

// Radiografias incluidas para poder demostrar el sistema sin depender de tener
// un archivo a mano. Ninguna pertenece al conjunto de entrenamiento.
const EJEMPLOS = [
  { etiqueta: "Caso con hallazgo", archivo: "/ejemplos/ejemplo-anomalia.jpeg" },
  { etiqueta: "Caso sin hallazgos", archivo: "/ejemplos/ejemplo-normal.jpeg" },
];

export default function Pagina() {
  const [imagen, setImagen] = useState<string | null>(null);
  const [nombreArchivo, setNombreArchivo] = useState("");
  const [momento, setMomento] = useState<string>("");
  const [resultado, setResultado] = useState<Respuesta | null>(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [arrastrando, setArrastrando] = useState(false);
  const entradaRef = useRef<HTMLInputElement>(null);

  // Se despierta el contenedor al cargar: se apaga tras un minuto sin trafico,
  // de modo que el primer analisis cargaria con el arranque en frio.
  useEffect(() => {
    fetch("/api/despertar").catch(() => {});
  }, []);

  const cargarArchivo = useCallback((archivo: File) => {
    setError(null);
    setResultado(null);

    if (!FORMATOS.includes(archivo.type)) {
      setError("Formato no admitido. Usa una imagen JPEG, PNG o WebP.");
      return;
    }
    if (archivo.size > TAMANO_MAXIMO) {
      setError(
        `La imagen pesa ${(archivo.size / 1024 / 1024).toFixed(1)} MB y el máximo es 8 MB.`,
      );
      return;
    }

    const lector = new FileReader();
    lector.onload = () => {
      setImagen(lector.result as string);
      setNombreArchivo(archivo.name);
    };
    lector.onerror = () => setError("No se pudo leer el archivo.");
    lector.readAsDataURL(archivo);
  }, []);

  async function cargarEjemplo(ruta: string, etiqueta: string) {
    setError(null);
    setResultado(null);
    try {
      const respuesta = await fetch(ruta);
      const blob = await respuesta.blob();
      const lector = new FileReader();
      lector.onload = () => {
        setImagen(lector.result as string);
        setNombreArchivo(etiqueta);
      };
      lector.readAsDataURL(blob);
    } catch {
      setError("No se pudo cargar la radiografía de ejemplo.");
    }
  }

  async function analizar() {
    if (!imagen) return;
    setCargando(true);
    setError(null);
    setResultado(null);

    try {
      const respuesta = await fetch("/api/clasificar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: imagen.split(",")[1] }),
      });
      const datos = await respuesta.json();

      if (!respuesta.ok) {
        setError(datos.detail ?? `Error ${respuesta.status} al analizar la imagen.`);
        return;
      }
      setResultado(datos as Respuesta);
      setMomento(
        new Date().toLocaleString("es", {
          dateStyle: "long",
          timeStyle: "short",
        }),
      );
    } catch {
      setError("No se pudo conectar con el servicio. Revisa tu conexión.");
    } finally {
      setCargando(false);
    }
  }

  function limpiar() {
    setImagen(null);
    setNombreArchivo("");
    setResultado(null);
    setError(null);
    if (entradaRef.current) entradaRef.current.value = "";
  }

  const esAnomalia = resultado?.prediction.label === "Anomaly";

  return (
    <div className="trama-fondo min-h-screen">
      {/* Cabecera */}
      <header className="sticky top-0 z-20 border-b border-[var(--borde)] bg-[var(--plano)]/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-3">
          <div className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className="grid h-8 w-8 place-items-center rounded-lg border border-[var(--acento)]/40 bg-[var(--acento)]/10"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 3v18M7 6.5v11M17 6.5v11M3.5 10v4M20.5 10v4"
                  stroke="var(--acento-vivo)"
                  strokeWidth="1.7"
                  strokeLinecap="round"
                />
              </svg>
            </span>
            <div className="leading-tight">
              <h1 className="text-sm font-semibold">Clasificación binaria radiológica</h1>
              <p className="text-[11px] text-[var(--tinta-tenue)]">
                Apoyo al diagnóstico mediante redes convolucionales
              </p>
            </div>
          </div>
          <span className="hidden rounded-full border border-[var(--borde)] px-2.5 py-1 font-mono text-[10px] text-[var(--tinta-media)] sm:block">
            YOLO11m-cls
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-8">
        {/* Carga */}
        {!resultado && (
          <section
            onDragOver={(e) => {
              e.preventDefault();
              setArrastrando(true);
            }}
            onDragLeave={() => setArrastrando(false)}
            onDrop={(e) => {
              e.preventDefault();
              setArrastrando(false);
              const archivo = e.dataTransfer.files?.[0];
              if (archivo) cargarArchivo(archivo);
            }}
            className={`rounded-2xl border-2 border-dashed p-8 text-center transition-colors sm:p-12 ${
              arrastrando
                ? "border-[var(--acento-vivo)] bg-[var(--acento)]/8"
                : "border-[var(--borde-fuerte)] bg-[var(--panel)]/55"
            }`}
          >
            <input
              ref={entradaRef}
              type="file"
              accept={FORMATOS.join(",")}
              className="hidden"
              onChange={(e) => {
                const archivo = e.target.files?.[0];
                if (archivo) cargarArchivo(archivo);
              }}
            />

            {imagen ? (
              <div className="space-y-5">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imagen}
                  alt="Radiografía seleccionada"
                  className="mx-auto max-h-64 rounded-xl border border-[var(--borde)] object-contain"
                />
                <p className="font-mono text-xs text-[var(--tinta-tenue)]">{nombreArchivo}</p>
                <div className="flex flex-wrap justify-center gap-3">
                  <button
                    onClick={analizar}
                    disabled={cargando}
                    className="rounded-lg bg-[var(--acento)] px-6 py-2.5 text-sm font-medium text-white shadow-lg shadow-[var(--acento)]/20 transition hover:bg-[var(--acento-vivo)] hover:text-[#04222b] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {cargando ? "Analizando…" : "Analizar radiografía"}
                  </button>
                  <button
                    onClick={limpiar}
                    disabled={cargando}
                    className="rounded-lg border border-[var(--borde-fuerte)] px-5 py-2.5 text-sm font-medium text-[var(--tinta-media)] transition hover:text-[var(--tinta)] disabled:opacity-50"
                  >
                    Elegir otra
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <span
                  aria-hidden="true"
                  className="mx-auto grid h-12 w-12 place-items-center rounded-xl border border-[var(--borde)] bg-[var(--panel-alto)]"
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <path
                      d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3"
                      stroke="var(--tinta-media)"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
                <div>
                  <p className="text-sm text-[var(--tinta-media)]">
                    Arrastra una radiografía de tórax o selecciónala desde tu equipo
                  </p>
                  <p className="mt-1 text-xs text-[var(--tinta-tenue)]">
                    JPEG, PNG o WebP · hasta 8 MB
                  </p>
                </div>
                <button
                  onClick={() => entradaRef.current?.click()}
                  className="rounded-lg bg-[var(--acento)] px-6 py-2.5 text-sm font-medium text-white shadow-lg shadow-[var(--acento)]/20 transition hover:bg-[var(--acento-vivo)] hover:text-[#04222b]"
                >
                  Seleccionar imagen
                </button>

                <div className="pt-3">
                  <p className="mb-2 text-xs text-[var(--tinta-tenue)]">
                    ¿No tienes una a mano? Prueba con estas:
                  </p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {EJEMPLOS.map(({ etiqueta, archivo }) => (
                      <button
                        key={archivo}
                        onClick={() => cargarEjemplo(archivo, etiqueta)}
                        className="rounded-lg border border-[var(--borde)] px-3 py-1.5 text-xs font-medium text-[var(--tinta-media)] transition hover:border-[var(--acento)]/50 hover:text-[var(--tinta)]"
                      >
                        {etiqueta}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        {cargando && !resultado && (
          <div className="mt-6 space-y-3">
            <div className="esqueleto h-56 rounded-xl" />
            <p className="text-center text-xs text-[var(--tinta-tenue)]">
              Si es el primer análisis tras un rato de inactividad, iniciar el servicio
              puede tardar unos segundos.
            </p>
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="mt-6 flex gap-3 rounded-xl border border-[var(--estado-critico)]/40 bg-[var(--estado-critico)]/10 p-4 text-sm"
          >
            <span aria-hidden="true" className="text-[var(--estado-critico)]">
              ⚠
            </span>
            <p className="text-[var(--tinta)]">{error}</p>
          </div>
        )}

        {/* Informe */}
        {resultado && imagen && (
          <article className="surgir space-y-5">
            {/* Encabezado del informe */}
            <header className="flex flex-wrap items-end justify-between gap-4 border-b border-[var(--borde)] pb-4">
              <div>
                <p className="text-[10px] font-medium uppercase tracking-[0.14em] text-[var(--tinta-tenue)]">
                  Informe de análisis
                </p>
                <h2 className="mt-1 font-mono text-sm text-[var(--tinta)]">
                  {nombreArchivo}
                </h2>
                <p className="mt-0.5 text-xs text-[var(--tinta-tenue)]">
                  {momento} · Modelo {resultado.performance.model_used}
                </p>
              </div>
              <div className="sin-imprimir flex gap-2">
                <button
                  onClick={() => window.print()}
                  className="rounded-lg border border-[var(--borde-fuerte)] px-4 py-2 text-xs font-medium text-[var(--tinta-media)] transition hover:text-[var(--tinta)]"
                >
                  Imprimir informe
                </button>
                <button
                  onClick={limpiar}
                  className="rounded-lg bg-[var(--acento)] px-4 py-2 text-xs font-medium text-white transition hover:bg-[var(--acento-vivo)] hover:text-[#04222b]"
                >
                  Analizar otra
                </button>
              </div>
            </header>

            <div className="grid gap-5 lg:grid-cols-[1.55fr_1fr]">
              {/* Visor */}
              <VisorRadiografia
                original={imagen}
                superpuesta={`data:image/jpeg;base64,${resultado.explainability.overlay_base64}`}
                mapa={`data:image/jpeg;base64,${resultado.explainability.heatmap_base64}`}
              />

              {/* Lateral */}
              <div className="space-y-5">
                {/* Resultado: el color nunca va solo, siempre con icono y texto */}
                <section
                  className={`rounded-xl border p-5 ${
                    esAnomalia
                      ? "border-[var(--estado-aviso)]/40 bg-[var(--estado-aviso)]/8"
                      : "border-[var(--estado-bien)]/40 bg-[var(--estado-bien)]/8"
                  }`}
                >
                  <p className="text-[10px] font-medium uppercase tracking-[0.14em] text-[var(--tinta-tenue)]">
                    Resultado
                  </p>
                  <div className="mt-2 flex items-center gap-2.5">
                    <span
                      aria-hidden="true"
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{
                        background: esAnomalia
                          ? "var(--estado-aviso)"
                          : "var(--estado-bien)",
                      }}
                    />
                    <p className="text-xl font-semibold text-[var(--tinta)]">
                      {esAnomalia ? "Hallazgo anómalo" : "Sin hallazgos"}
                    </p>
                  </div>
                  <div className="mt-5">
                    <MedidorConfianza
                      confianza={resultado.prediction.confidence}
                      umbral={UMBRAL_REVISION}
                    />
                  </div>
                </section>

                <PanelModelo />

                {/* Tiempos */}
                <section className="rounded-xl border border-[var(--borde)] bg-[var(--panel)] p-5">
                  <h2 className="mb-3 text-sm font-semibold">Tiempos de proceso</h2>
                  <dl className="space-y-2">
                    {(
                      [
                        ["Preprocesamiento", resultado.performance.preprocess_time_ms],
                        ["Inferencia", resultado.performance.inference_time_ms],
                        ["Grad-CAM", resultado.performance.explainability_time_ms],
                      ] as const
                    ).map(([etiqueta, valor]) => (
                      <div key={etiqueta} className="flex justify-between text-xs">
                        <dt className="text-[var(--tinta-media)]">{etiqueta}</dt>
                        <dd className="font-mono tabular-nums text-[var(--tinta)]">
                          {valor.toFixed(0)} ms
                        </dd>
                      </div>
                    ))}
                    <div className="flex justify-between border-t border-[var(--borde)] pt-2 text-xs">
                      <dt className="font-medium text-[var(--tinta)]">Total</dt>
                      <dd className="font-mono tabular-nums font-medium text-[var(--acento-vivo)]">
                        {resultado.performance.total_latency_ms.toFixed(0)} ms
                      </dd>
                    </div>
                  </dl>
                </section>
              </div>
            </div>
          </article>
        )}

        <footer className="mt-10 border-t border-[var(--borde)] pt-5">
          <p className="text-xs leading-relaxed text-[var(--tinta-tenue)]">
            <strong className="font-medium text-[var(--tinta-media)]">Aviso.</strong>{" "}
            Herramienta desarrollada con fines académicos como trabajo de grado en
            Ingeniería en Sistemas. No constituye un dispositivo médico ni sustituye el
            juicio de un profesional de la salud. Sus resultados no deben emplearse para
            tomar decisiones clínicas.
          </p>
        </footer>
      </main>
    </div>
  );
}
