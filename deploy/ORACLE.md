# Agentic OS 24/7 za darmo: Oracle Cloud Always Free

Ten poradnik stawia War Room na darmowym serwerze Oracle. Serwer działa 24/7, a z telefonu
otwierasz tylko stronę w przeglądarce. Na telefonie nic się nie instaluje.

Całość zajmuje około 20 minut klikania. Przygotowanie serwera robi skrypt
[`oracle-cloud-init.sh`](oracle-cloud-init.sh).

## 0. Zanim zaczniesz

- **Hasło do War Room.** Wymyśl długie hasło, co najmniej 12 znaków. Wpisujesz je tylko do skryptu w konsoli Oracle, nigdzie indziej.
- **Kod.** Repo `JavsonOf/agentic-os` jest prywatne, więc wybierz jedną z dwóch opcji:
  - **A (prostsza):** ustaw repo jako publiczne. W repo wejdź w Settings → General → na samym dole *Change visibility* → Public. W kodzie nie ma żadnych kluczy, klucze trzymasz w aplikacji.
  - **B:** utwórz token tylko do odczytu. Wejdź na GitHub → Settings → Developer settings → *Fine-grained tokens* → *Generate new token*. Ustaw:
    - Repository access: *Only select repositories* → `agentic-os`;
    - Permissions → Contents: *Read-only*.

    Token wkleisz do skryptu w polu `GITHUB_TOKEN`.
- **PR #2.** Jeśli PR #2 (War Room) nie jest jeszcze scalony, w skrypcie ustaw `BRANCH="claude/war-room"`. Po scaleniu wystarczy `main`.

## 1. Konto Oracle

1. Wejdź na <https://signup.cloud.oracle.com>.
2. Wybierz **Home Region** w Europie, np. *Germany Central (Frankfurt)*. Regionu domowego nie da się później zmienić.
3. Oracle poprosi o kartę. Pobiera tylko weryfikację, a darmowe zasoby (*Always Free*) nic nie kosztują.

## 2. Serwer (instancja)

1. W menu wybierz **Compute → Instances → Create instance**.
2. **Image and shape:**
   - Image: **Canonical Ubuntu 24.04** (albo 22.04).
   - Shape: *Change shape* → **Ampere** → **VM.Standard.A1.Flex**, a w nim **2 OCPU / 12 GB RAM**. To mieści się w darmowym limicie.
3. **Networking:** zostaw *Create new virtual cloud network* i upewnij się, że zaznaczone jest **Assign a public IPv4 address**.
4. **SSH keys:** zostaw *Generate a key pair* i pobierz klucz prywatny na wszelki wypadek. Do codziennego używania nie jest potrzebny.
5. Rozwiń **Show advanced options → Management → Initialization script → Paste cloud-init script** i wklej **cały** plik `oracle-cloud-init.sh`. Wcześniej zmień w nim:
   ```bash
   WAR_ROOM_PASSWORD="twoje-dlugie-haslo"
   BRANCH="main"            # albo "claude/war-room", jeśli PR #2 nie jest scalony
   GITHUB_TOKEN=""          # tylko opcja B (prywatne repo)
   ```
6. Kliknij **Create**.

Jeśli pojawi się **"Out of capacity"**, ARM w tym miejscu jest chwilowo zajęty. Spróbuj:
- innej *Availability Domain*;
- mniejszego shape'u (1 OCPU / 6 GB);
- ponownie za kilka godzin.

## 3. Otwórz porty 80 i 443 (firewall Oracle)

1. Na stronie instancji kliknij podsieć (*Subnet*), potem **Security Lists → Default Security List**.
2. Kliknij **Add Ingress Rules** i dodaj dwie reguły:
   - Source CIDR `0.0.0.0/0`, IP Protocol TCP, Destination Port `80`;
   - Source CIDR `0.0.0.0/0`, IP Protocol TCP, Destination Port `443`.

Firewall wewnątrz samego serwera skrypt otwiera sam.

## 4. Gotowe: otwórz War Room na telefonie

1. Odczekaj 5–10 minut po starcie instancji. Skrypt w tym czasie instaluje Dockera, buduje aplikację i pobiera certyfikat HTTPS.
2. Skopiuj **Public IP address** ze strony instancji, np. `130.61.12.34`.
3. Otwórz w telefonie adres z kropkami zamienionymi na myślniki:
   **`https://130-61-12-34.sslip.io`**
4. Zaloguj się hasłem `WAR_ROOM_PASSWORD`.
5. W zakładce **🔑 Modele** dodaj klucz Gemini (lub innego dostawcy).
6. W menu przeglądarki wybierz **"Dodaj do ekranu głównego"**. War Room działa wtedy jak aplikacja.

## Jak to działa i o co zadbano

- **Bezpieczeństwo:**
  - aplikacja nasłuchuje tylko lokalnie (`127.0.0.1:8000`), a z zewnątrz dostęp prowadzi wyłącznie przez Caddy z HTTPS (darmowy certyfikat Let's Encrypt);
  - każde API wymaga hasła;
  - kontener działa jako zwykły użytkownik, nie root;
  - klucze API trzymane są w plikach z prawami `600`.
- **Dane** (agenci, klucze, pamięć, MCP) leżą w `/opt/agentic-os/data` i przetrwają restart oraz aktualizację.
- **Aktualizacje:** co 15 minut serwer sprawdza GitHub. Gdy na gałęzi pojawi się nowy commit, sam przebudowuje i restartuje aplikację.
- **Restart serwera:** kontenery wstają same (`--restart unless-stopped`).

## Problemy

- **Strona się nie otwiera.**
  - Sprawdź reguły z kroku 3.
  - Sprawdź log instalacji: w konsoli Oracle otwórz instancję → *Console connection* (albo SSH: `ssh -i klucz.key ubuntu@IP`) i uruchom:
    ```bash
    sudo tail -50 /var/log/agentic-os-setup.log
    ```
- **Zmiana hasła:** zaloguj się przez SSH, popraw `AGENTIC_OS_API_KEY` w `/opt/agentic-os/.env`, potem uruchom `sudo /opt/agentic-os/deploy.sh --force`.
- **Uwaga o widoczności sekretów:** skrypt cloud-init (z hasłem i ewentualnym tokenem) widzi w konsoli Oracle każdy, kto ma dostęp do Twojego konta Oracle. Nie udostępniaj konta.
- **Bezczynna instancja:** Oracle może odebrać darmową instancję, która przez 7 dni prawie nic nie robi (bardzo niskie CPU). Gdyby to się stało, możesz przejść na konto *Pay As You Go*. Darmowe limity dalej obowiązują i nic nie płacisz, dopóki się w nich mieścisz, a instancje nie są wtedy odbierane.
