import { NextResponse } from "next/server";

/**
 * Despierta el contenedor de inferencia.
 *
 * El servicio se apaga tras un minuto sin trafico para no consumir credito,
 * de modo que la primera peticion tras un periodo de inactividad tarda unos
 * once segundos en lugar de dos. La interfaz llama a esta ruta al cargarse,
 * de manera que el contenedor ya este activo cuando la persona termine de
 * seleccionar la radiografia.
 */

export const maxDuration = 60;

export async function GET() {
  const servicio = process.env.XRAY_API_URL;

  if (!servicio) {
    return NextResponse.json({ listo: false, motivo: "sin configurar" }, { status: 200 });
  }

  try {
    const respuesta = await fetch(`${servicio.replace(/\/$/, "")}/health`, {
      signal: AbortSignal.timeout(50_000),
      cache: "no-store",
    });
    return NextResponse.json({ listo: respuesta.ok }, { status: 200 });
  } catch {
    return NextResponse.json({ listo: false }, { status: 200 });
  }
}
