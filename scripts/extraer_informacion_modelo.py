#!/usr/bin/env python3
"""
Script para extraer toda la información del modelo NeuroRiskLogic:
- Lista de preguntas (con corrección de "el paciente" a primera persona)
- Pesos de cada pregunta
- Factores de riesgo y protección
- Recomendaciones posibles
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.utils.feature_definitions import load_feature_definitions
from app.utils.risk_calculator import RiskCalculator
from app.models.predictor import NeuroriskPredictor


def corregir_pregunta(pregunta):
    """Convierte preguntas de tercera persona a primera persona."""
    # Mapeo de correcciones
    correcciones = {
        "Are the subject's parents blood-related?": "¿Mis padres son parientes consanguíneos?",
        "Does the subject have a family history of neurological disorders?": (
            "¿Tengo antecedentes familiares de trastornos neurológicos o psiquiátricos?"
        ),
        "Has the subject ever had seizures or convulsions?": (
            "¿He tenido alguna vez convulsiones o crisis epilépticas?"
        ),
        "History of traumatic brain injury (TBI) or head trauma?": (
            "¿He tenido alguna lesión cerebral traumática o trauma craneal?"
        ),
        "Diagnosed with any psychiatric disorder?": (
            "¿He sido diagnosticado/a con algún trastorno psiquiátrico?"
        ),
        "Does the subject consume psychoactive substances regularly?": (
            "¿Consumo sustancias psicoactivas regularmente?"
        ),
        "History of suicide attempts or ideation?": (
            "¿He tenido intentos de suicidio o ideación suicida?"
        ),
        "Is the subject on psychotropic or neurological medication?": (
            "¿Estoy tomando medicación psicotrópica o neurológica?"
        ),
        "Was the subject born with complications or low weight?": (
            "¿Nací con complicaciones o bajo peso?"
        ),
        "Does the subject live in extreme poverty?": ("¿Vivo en condiciones de pobreza extrema?"),
        "Has the subject ever had access restrictions to education?": (
            "¿He tenido restricciones de acceso a la educación?"
        ),
        "Does the subject have access to mental healthcare?": (
            "¿Tengo acceso a servicios de salud mental?"
        ),
        "Has the subject ever been diagnosed with a disability?": (
            "¿He sido diagnosticado/a con alguna discapacidad?"
        ),
        "Was the subject breastfed during infancy?": ("¿Fui amamantado/a durante la infancia?"),
        "Exposure to violence or trauma (childhood or adulthood)?": (
            "¿He estado expuesto/a a violencia o trauma (infancia o adultez)?"
        ),
    }

    return correcciones.get(pregunta, pregunta)


def extraer_toda_informacion():
    """Extrae toda la información del modelo."""

    # 1. Cargar definiciones
    feature_defs = load_feature_definitions()
    calculator = RiskCalculator()

    # 2. Información de las preguntas
    print("=" * 80)
    print("CALCULADORA DE RIESGO DE ALTERACIÓN EN LA SALUD MENTAL")
    print("=" * 80)
    print("\n## 1. LISTA DE PREGUNTAS Y SUS PESOS\n")

    preguntas_info = []

    for feature in feature_defs.features.values():
        if feature["name"] in ["age", "gender"]:
            continue  # Skip demographic fields

        pregunta_info = {
            "id": feature.get("id", 0),
            "campo": feature["name"],
            "pregunta_original": feature.get("description", ""),
            "pregunta_corregida": corregir_pregunta(feature.get("description", "")),
            "tipo": feature["type"],
            "categoria": feature["category"],
            "peso": feature.get("weight", 0),
            "direccion_riesgo": feature.get("risk_direction", "unknown"),
            "relevancia_clinica": feature.get("relevance", ""),
        }
        preguntas_info.append(pregunta_info)

    # Ordenar por peso descendente
    preguntas_info.sort(key=lambda x: x["peso"], reverse=True)

    print("### Preguntas ordenadas por peso (impacto en el riesgo):\n")
    for i, p in enumerate(preguntas_info, 1):
        print(f"{i}. **{p['pregunta_corregida']}**")
        print(f"   - Campo técnico: {p['campo']}")
        print(f"   - Peso: {p['peso']:.2%}")
        factor_type = (
            "Factor de Riesgo" if p["direccion_riesgo"] == "positive" else "Factor Protector"
        )
        print(f"   - Tipo: {factor_type}")
        print(f"   - Relevancia: {p['relevancia_clinica']}")
        print()

    # 3. Factores de riesgo y protección
    print("\n" + "=" * 80)
    print("## 2. FACTORES DE RIESGO Y PROTECCIÓN\n")

    factores_riesgo = []
    factores_protectores = []

    for feature_name, feature_info in feature_defs.features.items():
        if feature_info.get("risk_direction") == "positive":
            factores_riesgo.append(
                {
                    "nombre": feature_info["display_name"],
                    "campo": feature_name,
                    "peso": feature_info.get("weight", 0),
                }
            )
        elif feature_info.get("risk_direction") == "negative":
            factores_protectores.append(
                {
                    "nombre": feature_info["display_name"],
                    "campo": feature_name,
                    "peso": feature_info.get("weight", 0),
                }
            )

    print("### FACTORES DE RIESGO (aumentan la probabilidad):")
    for f in sorted(factores_riesgo, key=lambda x: x["peso"], reverse=True):
        print(f"- {f['nombre']} (peso: {f['peso']:.2%})")

    print("\n### FACTORES PROTECTORES (disminuyen la probabilidad):")
    for f in sorted(factores_protectores, key=lambda x: x["peso"], reverse=True):
        print(f"- {f['nombre']} (peso: {f['peso']:.2%})")

    # 4. Posibles recomendaciones
    print("\n" + "=" * 80)
    print("## 3. RECOMENDACIONES SEGÚN NIVEL DE RIESGO\n")

    # Simular diferentes escenarios para ver recomendaciones
    escenarios = {
        "Riesgo Bajo": {
            "family_neuro_history": False,
            "psychiatric_diagnosis": False,
            "suicide_ideation": False,
            "healthcare_access": True,
            "social_support_level": "supported",
        },
        "Riesgo Moderado": {
            "family_neuro_history": True,
            "psychiatric_diagnosis": True,
            "suicide_ideation": False,
            "healthcare_access": True,
            "social_support_level": "moderate",
        },
        "Riesgo Alto": {
            "family_neuro_history": True,
            "psychiatric_diagnosis": True,
            "suicide_ideation": True,
            "healthcare_access": False,
            "social_support_level": "isolated",
        },
    }

    recomendaciones_todas = set()

    for nivel, datos in escenarios.items():
        # Completar datos mínimos
        datos_completos = {
            "age": 30,
            "gender": "M",
            "consanguinity": False,
            "seizures_history": False,
            "brain_injury_history": False,
            "substance_use": False,
            "psychotropic_medication": False,
            "birth_complications": False,
            "extreme_poverty": False,
            "education_access_issues": False,
            "disability_diagnosis": False,
            "breastfed_infancy": True,
            "violence_exposure": False,
        }
        datos_completos.update(datos)

        # Generar recomendaciones
        risk_factors = []
        if datos.get("family_neuro_history"):
            risk_factors.append("Family history")
        if datos.get("psychiatric_diagnosis"):
            risk_factors.append("Psychiatric diagnosis")
        if datos.get("suicide_ideation"):
            risk_factors.append("Suicide ideation")

        recomendaciones = calculator.generate_recommendations(
            nivel.lower().replace(" ", "_"), risk_factors, datos_completos
        )

        print(f"### {nivel}:")
        for rec in recomendaciones:
            print(f"- {rec}")
            recomendaciones_todas.add(rec)
        print()

    # 5. Tabla resumen
    print("\n" + "=" * 80)
    print("## 4. TABLA RESUMEN DE IMPACTO POR PREGUNTA\n")

    print("| Pregunta | Peso | Tipo | Activa Factor |")
    print("|----------|------|------|---------------|")

    for p in preguntas_info:
        tipo = "Riesgo" if p["direccion_riesgo"] == "positive" else "Protector"
        print(f"| {p['pregunta_corregida'][:50]}... | {p['peso']:.1%} | {tipo} | {p['campo']} |")

    # 6. Exportar a JSON
    resultado_completo = {
        "nombre_sistema": "Calculadora de Riesgo de Alteración en la Salud Mental",
        "version": "1.0.0",
        "preguntas": preguntas_info,
        "factores_riesgo": factores_riesgo,
        "factores_protectores": factores_protectores,
        "recomendaciones_posibles": list(recomendaciones_todas),
        "umbrales_riesgo": {"bajo": "0.0 - 0.3", "moderado": "0.3 - 0.7", "alto": "0.7 - 1.0"},
    }

    # Guardar JSON
    with open("informacion_modelo_salud_mental.json", "w", encoding="utf-8") as f:
        json.dump(resultado_completo, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Información exportada a: informacion_modelo_salud_mental.json")

    # 7. Nota sobre factores vacíos
    print("\n## NOTA IMPORTANTE:")
    print("Cuando no hay factores de riesgo o protección, el sistema debe mostrar:")
    print('- Si no hay factores de riesgo: "No se identificaron factores de riesgo"')
    print('- Si no hay factores protectores: "No se identificaron factores protectores"')


if __name__ == "__main__":
    extraer_toda_informacion()
