/*  
    Autor:   Matej Poljuha
    Kolegij: Paralelno programiranje na heterogenim sustavima
    Tema:    Izračun vrijednosti određenog integrala trapeznom formulom - CPU-only varijanta
*/

#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace std;

double f (double);
int valid_segment(double, double, long);
void progress(double);
double trapeze_formula (double, double, long);
void benchmarking (chrono::_V2::high_resolution_clock::time_point, chrono::_V2::high_resolution_clock::time_point);

int main (void){
    long n;                                                                 // Broj podsegmenata
    double a, b;                                                            // Rubne točke početnog segmenta
    
    cout << "Podintegralna funkcija: " << "sin(x)" << endl;

    do {
        cout << "\nUnos granica integracije,\na = ";                        // Zadavanje parametara
        cin >> a;                                                           // Donja granica integrala
        cout << "b = ";
        cin >> b;                                                           // Gornja granica integrala
        cout << "Unos broja segmenata\nn = ";
        cin >> n;                                                           // Broj segmenata za trapeznu formulu
    } while (valid_segment(a, b, n));
    
    /*
        Računanje vrijednosti integrala (poziv funkcije)
    */
    
    auto start = chrono::high_resolution_clock::now();                      // Početak mjerenja vremena izvođenja
    cout << "\nDobivena vrijednost integrala je: " << std::fixed << std::setprecision(80) << endl; // Poziv funkcije računanja trapezne formule
    cout << trapeze_formula(a, b, n) << endl;
    auto end = chrono::high_resolution_clock::now();                        // Kraj mjerenja vremena izvođenja
    
    /*
        Ispis vremena izvođenja
    */
    cout << "\nUKUPNO VRIJEME IZVOĐENJA:" << endl;
    benchmarking(start, end);
    
    return 0;
}

double f (double x){
    /*
        Podintegralna funkcija
    */

    return sin(x);
}

int valid_segment(double a, double b, long n){
    /*
        Priručna funkcija za provjeru valjanosti početnih podataka (funkcionalna samo za neke funkcije, detektira jedino intervale prekinutosti > 0.001)
    */

    double i = a;
    if (a >= b){                                                            // Prvi uneseni broj (a) se smatra početkom segmenta i mora biti manji od broja b
        cout << "\nPočetak segmenta mora biti manji od kraja segmenta." << endl;
        return 1;
    }
    while (i <= b){                                                         // Funkcija mora biti neprekinuta na segmentu
        if(isnan(f(i))){
            cout << "\nFunkcija nije neprekinuta na segmentu, ponovi unos." << endl;
            return 1;
        }
        else{
            i += 0.001;
        }
    }
    if (n < 0){                                                             // Broj segmenata mora biti pozitivan broj
        cout << "\nBroj segmenata mora biti pozitivan cijeli broj." << endl;
        return 1;
    }
    return 0;
}

void progress(double progress){
    /*
        Funkcija za ispis progress bar-a (prisutni vizualni glitchevi na windows platformi)
    */

    int barLength = 80;
    int pos = progress * barLength;

    if (pos == barLength){                                                  // If koji uklanja progress bar nakon što se program izvrši
        cout << "\33[2K\r";
        cout.flush();
        return;
    }

    cout<<" Izračun: [";
    for(int i=0; i < barLength; ++i){                                       // Ispis progress bar-a
        if(i < pos)
            cout << "=";
        else if(i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << int(progress * 100) << " %\r";
    cout.flush();
}

double trapeze_formula (double a, double b, long n){
    /*  
        Funkcija trapezne formule:       I* = h/2 * (y0 + 2y1 + 2y2 + ... + 2yn-1 + yn)
    */
    
    double sum = 0;                                                         // Izračunata aproksimacija
    double x,y;                                                             // Koordinate rubnih točaka svakog podsegmenta u obliku vektora
    double h = (b-a)/n;                                                     // Duljina podsegmenta

    for (int i = 0; i < n+1; i++){
        x = a + i*h;                                                        // x koordinate se računaju kao zbroj x koordinate donje granice integrala (a) + redni broj segmenta (i) * duljina segmenta
        y = f(x);                                                           // y koordinate se računaju funkcijom f

        if (i == 0 || i == n){
            sum += (h/2.0) * y;                                             // Formula za prvi i zadnji element
        }
        else{
            sum += h*y;                                                     // Formula za sve ostale elemente
        }
        
        if(i % 100000 == 0 || i == n){                                        // Poziv progress bar funkcije, napredak se ažurira u svakoj 100000. iteraciji jer je program izvršen gotovo instantno za manje od 100000 segmenata
            //progress(double(i)/n);
        }
    }
    return sum;
}

void benchmarking (chrono::_V2::high_resolution_clock::time_point start, chrono::_V2::high_resolution_clock::time_point end){
    using namespace chrono;

    /*
        Točno vrijeme u mikrosekundama
    */

    auto exact_vrijeme = duration_cast<microseconds>(end - start);                                      // Točno vrijeme izvođenja u mikrosekundama
    cout << "Vrijeme izvođenja:            " << exact_vrijeme.count() << " mikrosekundi" << endl;       // Ispis točnog vremena izvođenja u mikrosekundama

    /*
        Human readable vrijeme
    */

    cout << "Human readable format:        ";

    auto minute = duration_cast<minutes>(exact_vrijeme);                    // Iz vremena u mikrosekundama uzima broj sekundi
    cout << minute.count() << " min ";
    exact_vrijeme -= duration_cast<microseconds>(minute);                   // Vrijeme u sekundama pretvara natrag u mikrosekunde i oduzima od ukupnog vremena u mikrosekundama

    auto sekunde = duration_cast<seconds>(exact_vrijeme);
    cout << sekunde.count() << " s ";
    exact_vrijeme -= duration_cast<microseconds>(sekunde);

    auto milisekunde = duration_cast<milliseconds>(exact_vrijeme);
    cout << milisekunde.count() << " ms ";
    exact_vrijeme -= duration_cast<microseconds>(milisekunde);

    cout << exact_vrijeme.count() << " μs" << endl;
}