import { NextResponse } from "next/server";

/**
 * Intermediario entre el navegador y el servicio de inferencia.
 *
 * La URL del servicio se lee de una variable de entorno del servidor, de modo
 * que nunca llega al navegador: el cliente solo conoce esta ruta. Asi se evita
 * que terceros extraigan el endpoint del codigo del cliente y consuman el
 * credito de la cuenta, y de paso las peticiones quedan en el mismo origen,
 * con lo que no interviene CORS.
 */

// El arranque en frio del contenedor de inferencia ronda los 11 segundos, por
// encima del limite por defecto de las funciones serverless.
export const maxDuration = 60;

export async function POST(request: Request) {
  const servicio = process.env.XRAY_API_URL;

  if (!servicio) {
    return NextResponse.json(
      { detail: "El servicio de inferencia no esta configurado (falta XRAY_API_URL)." },
      { status: 500 },
    );
  }

  let cuerpo: unknown;
  try {
    cuerpo = await request.json();
  } catch {
    return NextResponse.json({ detail: "Peticion malformada." }, { status: 400 });
  }

  try {
    const respuesta = await fetch(`${servicio.replace(/\/$/, "")}/cnn_xray_demo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cuerpo),
      signal: AbortSignal.timeout(55_000),
    });

    const datos = await respuesta.json();
    return NextResponse.json(datos, { status: respuesta.status });
  } catch (error) {
    const esTiempoAgotado = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      {
        detail: esTiempoAgotado
          ? "El servicio tardo demasiado en responder. Vuelve a intentarlo: el primer analisis tras un periodo de inactividad requiere iniciar el contenedor."
          : "No se pudo contactar con el servicio de inferencia.",
      },
      { status: 504 },
    );
  }
}
