/*  
    Autor:   Matej Poljuha
    Kolegij: Paralelno programiranje na heterogenim sustavima
    Tema:    Izračun vrijednosti određenog integrala trapeznom formulom - CPU varijanta
*/

#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <vector>
#include <typeinfo>

using namespace std;

double f (double x){                                                    // Podintegralna funkcija
    return abs(sin(pow(x,x))/pow(2,(pow(x,x) - M_PI/2)/M_PI));
}

void tocke (vector<double> &x, vector<double> &y, double a, int n, double h){
    for (int i = 0; i<n+1; i++){                                        // Petlja u kojoj se računaju koordinate točaka za korištenje u trapeznoj formuli
        x.push_back(a + i*h);                                           // x[i] vrijednosti se računaju kao zbroj x koordinate donje granice integrala (a) + redni broj segmenta (i) * duljina segmenta
        y.push_back(f(x[i]));                                           // y[i] vrijednosti se računaju funkcijom f s ulaznim parametrom x[i]
    }
}

double trapezna_formula (double a, double b, int n){
    /*  
        Trapezna formula:       I* = h/2 * (y0 + 2y1 + 2y2 + ... + 2yn-1 + yn)
        Razdvojeno u 2 koraka:
            (1) petlja koja računa h/2 * (2y1 + 2y2 + ... + 2yn-1)
            (2) izraz koji zbraja vrijednost iz koraka 1 i h/2 * (y0 + yn)
    */
    
    double sum = 0;
    vector<double> x,y;                                                 // Koordinate točaka zapisane u obliku vektora
    double h = (b-a)/n;                                                 // Duljina segmenta

    tocke(x, y, a, n, h);

    for (int i = 1; i<n; i++){                                          // korak (1)
        sum = sum + h*y[i];
    }

    return (h/2.0) * (y[0] + y[n]) + sum;                               // korak (2) - aproksimirana vrijednost određenog integrala
}

void benchmarking (chrono::_V2::high_resolution_clock::time_point start, chrono::_V2::high_resolution_clock::time_point end){
    using namespace chrono;
    cout << "\nTIMING:" << endl;

    /*
        Točno vrijeme u mikrosekundama
    */

    auto exact_vrijeme = duration_cast<microseconds>(end - start);                              // Točno vrijeme izvođenja u mikrosekundama
    cout << "Vrijeme izvođenja:     " << exact_vrijeme.count() << " mikrosekundi" << endl;      // Ispis točnog vremena izvođenja u mikrosekundama

    /*
        Human readable vrijeme
    */

    cout << "Human readable format: ";

    auto minute = duration_cast<minutes>(exact_vrijeme);                        // Iz vremena u mikrosekundama uzima broj sekundi
    cout << minute.count() << " min ";
    exact_vrijeme -= duration_cast<microseconds>(minute);                       // Vrijeme u sekundama pretvara natrag u mikrosekunde i oduzima od ukupnog vremena u mikrosekundama

    auto sekunde = duration_cast<seconds>(exact_vrijeme);
    cout << sekunde.count() << " s ";
    exact_vrijeme -= duration_cast<microseconds>(sekunde);

    auto milisekunde = duration_cast<milliseconds>(exact_vrijeme);
    cout << milisekunde.count() << " ms ";
    exact_vrijeme -= duration_cast<microseconds>(milisekunde);

    cout << exact_vrijeme.count() << " μs" << endl;
}

int main (void){
    int n;
    double a, b;

    cout << "Podintegralna funkcija: " << "|sin(x**x)/2**((x**x-pi/2)/pi)|" << endl;

    cout << "Unos granica integracije,\na = ";                          // Zadavanje parametara
    cin >> a;                                                           // Donja granica integrala
    cout << "b = ";
    cin >> b;                                                           // Gornja granica integrala
    cout << "Unos broja segmenata\nn = ";
    cin >> n;                                                           // Broj segmenata za trapeznu formulu
    
    /*
        Računanje vrijednosti integrala (poziv funkcije)
    */
    /*
    auto start = chrono::high_resolution_clock::now();
    cout << "\nDobivena vrijednost integrala je: " << std::fixed << std::setprecision(35) << trapezna_formula(a, b, n) << endl;    // Poziv funkcije računanja trapezne formule
    auto end = chrono::high_resolution_clock::now();
    */
    /*
        Ispis vremena izvođenja
    */
    
    //benchmarking(start, end);
    
    

    return 0;
}