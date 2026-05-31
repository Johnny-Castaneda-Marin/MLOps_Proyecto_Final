build-streamlit:
    runs-on: ubuntu-latest
    needs: detect-changes
    if: needs.detect-changes.outputs.streamlit == 'true' || github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract short SHA
        id: sha
        run: echo "short_sha=${GITHUB_SHA::7}" >> "$GITHUB_OUTPUT"

      - name: Build and push Streamlit image
        uses: docker/build-push-action@v6
        with:
          # Context es la raíz del repo porque el Dockerfile copia streamlit/ desde allí
          context: .
          file: streamlit/Dockerfile
          push: true
          tags: |
            ${{ env.IMAGE_PREFIX }}-streamlit:${{ steps.sha.outputs.short_sha }}
            ${{ env.IMAGE_PREFIX }}-streamlit:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
