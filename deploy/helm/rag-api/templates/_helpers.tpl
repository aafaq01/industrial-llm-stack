{{/* Env shared by the API, the migrate hook and the ingest Job (same as the compose rag-env anchor). */}}
{{- define "rag.env" -}}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef: {name: {{ .Values.existingSecret }}, key: POSTGRES_PASSWORD}
- name: ELASTIC_PASSWORD
  valueFrom:
    secretKeyRef: {name: {{ .Values.existingSecret }}, key: ELASTIC_PASSWORD}
- name: DATABASE_URL
  value: postgresql+psycopg://postgres:$(POSTGRES_PASSWORD)@{{ .Values.env.POSTGRES_HOST }}:5432/postgres
- {name: ES_URL, value: {{ .Values.env.ES_URL | quote }}}
- {name: LLM_BASE_URL, value: {{ .Values.env.LLM_BASE_URL | quote }}}
{{- end }}

{{/* Pod-level bits shared by all three workloads. */}}
{{- define "rag.podSpec" -}}
automountServiceAccountToken: false
securityContext: {{- toYaml .Values.podSecurityContext | nindent 2 }}
volumes:
  - name: tmp
    emptyDir: {}
  - name: corpus
    emptyDir: {}
{{- end }}

{{- define "rag.mounts" -}}
volumeMounts:
  - {name: tmp, mountPath: /tmp}
  - {name: corpus, mountPath: /data/corpus}
securityContext: {{- toYaml .Values.containerSecurityContext | nindent 2 }}
{{- end }}
